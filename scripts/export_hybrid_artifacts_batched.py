"""GPU-optimised batch export for hybrid Qdrant artifacts.

This script is functionally equivalent to ``export_hybrid_artifacts.py``
but restructures the pipeline so that **chunks from many documents are
collected into large batches** before calling the dense-embedding model.

Why this matters
~~~~~~~~~~~~~~~~
The original script embeds one document at a time.  A typical legal
document may only produce 5-20 chunks, so even with ``batch_size=32``
the GPU is mostly idle.  Here we decouple chunking (CPU) from
embedding (GPU) so we can fill GPU batches to the brim, regardless of
how many documents the chunks come from.

Workflow
~~~~~~~~
1. **Chunking phase** – iterate the CSV once, parse → tree → chunks.
   Chunks are accumulated in an in-memory buffer.  Every time the
   buffer reaches *gpu_batch_size*, we flush it through the GPU.
2. **Flush** – a flush encodes the buffer, builds ``PointStruct``s with
   both dense and sparse vectors, writes them to the artifact sink, and
   records per-document checkpoint entries.
3. **Final flush** – any remaining chunks are flushed at the end.

Because checkpoints are written per-document *after* the document's
chunks are fully flushed, crash-recovery semantics remain identical.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings  # noqa: E402
from app.services.chunking.html_to_markdown import html_to_markdown  # noqa: E402
from app.services.chunking.legal_chunk_extractor import ChunkingConfig, LegalTreeChunkExtractor  # noqa: E402
from app.services.chunking.legal_tree_builder import build_document_tree  # noqa: E402
from app.services.retrieval.artifact_store import ArtifactPointSink  # noqa: E402
from app.services.retrieval.checkpoint_store import SQLiteCheckpointStore  # noqa: E402
from app.services.retrieval.hybrid_support import (  # noqa: E402
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    DEFAULT_PIPELINE_VERSION,
    BM25_MODEL_NAME,
    DEFAULT_BM25_OPTIONS,
    SentenceTransformerDenseEncoder,
    build_bm25_document,
    build_chunk_payload,
    deterministic_point_id,
)

from qdrant_client import models  # noqa: E402


# ---------------------------------------------------------------------------
# Pending-chunk bookkeeping
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PendingChunk:
    """A chunk waiting to be embedded, with back-references to its document."""

    doc_id: str
    payload: dict[str, Any]
    context_text: str


@dataclass(slots=True)
class DocumentBatch:
    """Tracks all pending chunks that belong to a single document.

    Once *all* chunks for a document have been flushed we can mark the
    document as ``done`` in the checkpoint store.
    """

    doc_id: str
    total_chunks: int = 0
    flushed_chunks: int = 0

    @property
    def is_complete(self) -> bool:
        return self.flushed_chunks >= self.total_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "GPU-optimised hybrid artifact export. "
            "Collects chunks from multiple documents into large "
            "GPU batches to maximise throughput."
        ),
    )
    parser.add_argument("--content-csv", default="data/content.csv", help="CSV with id,content_html columns.")
    parser.add_argument("--metadata-csv", default="data/metadata.csv", help="CSV with document metadata.")
    parser.add_argument("--relationships-csv", default="data/relationships.csv", help="CSV with document relationships.")
    parser.add_argument("--output-dir", default="data/artifacts/hybrid_qdrant_batched", help="Directory for manifest and shard files.")
    parser.add_argument("--dense-model-name", default=settings.dense_embedding_model, help="Hugging Face/SentenceTransformers model.")
    parser.add_argument("--sparse-model-name", default=settings.sparse_embedding_model, help="Sparse model name for local Qdrant BM25 import.")
    parser.add_argument("--bm25-k", type=float, default=settings.bm25_k, help="BM25 k parameter.")
    parser.add_argument("--bm25-b", type=float, default=settings.bm25_b, help="BM25 b parameter.")
    parser.add_argument("--bm25-language", default=settings.bm25_language, help="BM25 language; use none for Vietnamese.")
    parser.add_argument("--collection", default=settings.qdrant_collection_hybrid, help="Target Qdrant collection name.")
    parser.add_argument("--device", default=settings.dense_embedding_device, help="Embedding device (cuda, cpu).")
    parser.add_argument("--embed-batch-size", type=int, default=settings.dense_embedding_batch_size, help="Inner batch size passed to SentenceTransformer.encode().")
    parser.add_argument("--gpu-batch-size", type=int, default=256, help="Number of chunks to collect before flushing through the GPU. Larger = better GPU utilisation.")
    parser.add_argument("--shard-size", type=int, default=10_000, help="Number of points per artifact shard.")
    parser.add_argument("--checkpoint-db", default=settings.ingest_checkpoint_db, help="SQLite checkpoint path.")
    parser.add_argument("--dead-letter", default="data/processed/hybrid_artifact_batched_dead_letter.jsonl", help="JSONL file for failed rows.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint DB.")
    parser.add_argument("--limit", type=int, default=None, help="Limit CSV rows for dry run.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Approximate chunk size in tokens.")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Overlap for long text chunks.")
    parser.add_argument("--table-chunk-size", type=int, default=500, help="Approximate chunk size for tables.")
    parser.add_argument("--pipeline-version", default=settings.ingest_pipeline_version, help="Deterministic point-id version prefix.")
    parser.add_argument("--skip-sidecar-import", action="store_true", help="Skip CSV metadata import into SQLite.")
    return parser


# ---------------------------------------------------------------------------
# Batched pipeline
# ---------------------------------------------------------------------------

class BatchedHybridExportPipeline:
    """Decouples CPU-bound chunking from GPU-bound embedding.

    Instead of embedding per-document we accumulate a buffer of
    ``PendingChunk`` items and flush them through the GPU in one shot
    whenever the buffer reaches ``gpu_batch_size``.
    """

    def __init__(
        self,
        *,
        checkpoint_store: SQLiteCheckpointStore,
        dense_encoder: SentenceTransformerDenseEncoder,
        artifact_sink: ArtifactPointSink,
        content_csv: Path,
        metadata_csv: Path,
        relationships_csv: Path,
        dead_letter_path: Path,
        collection_name: str,
        pipeline_version: str,
        sparse_model_name: str,
        bm25_options: dict[str, Any],
        gpu_batch_size: int = 256,
        shard_size: int = 10_000,
        resume: bool = False,
        limit: int | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 64,
        table_chunk_size: int = 500,
        import_sidecar: bool = True,
    ) -> None:
        self.checkpoint_store = checkpoint_store
        self.dense_encoder = dense_encoder
        self.artifact_sink = artifact_sink

        self.content_csv = content_csv
        self.metadata_csv = metadata_csv
        self.relationships_csv = relationships_csv
        self.dead_letter_path = dead_letter_path
        self.collection_name = collection_name
        self.pipeline_version = pipeline_version
        self.sparse_model_name = sparse_model_name
        self.bm25_options = bm25_options
        self.gpu_batch_size = max(1, gpu_batch_size)
        self.shard_size = shard_size
        self.resume = resume
        self.limit = limit
        self.import_sidecar = import_sidecar

        self.extractor = LegalTreeChunkExtractor(
            ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                table_chunk_size=table_chunk_size,
                include_relationships=False,
            )
        )

        self.run_id = f"bat_{uuid.uuid4().hex[:12]}"

        # Pending-chunk buffer
        self._pending: list[PendingChunk] = []
        # doc_id → DocumentBatch mapping (only for docs with pending chunks)
        self._doc_batches: dict[str, DocumentBatch] = {}

        self.stats: dict[str, Any] = {
            "docs_seen": 0,
            "docs_processed": 0,
            "docs_skipped": 0,
            "docs_failed": 0,
            "docs_empty": 0,
            "chunks_total": 0,
            "chunks_upserted": 0,
            "gpu_flushes": 0,
            "total_embed_time_s": 0.0,
        }

    # ----- public entry point -----

    async def run(self) -> dict[str, Any]:
        self._configure_csv_limits()
        self.checkpoint_store.init_schema()
        if self.import_sidecar:
            self.checkpoint_store.import_metadata_csv(self.metadata_csv)
            self.checkpoint_store.import_relationships_csv(self.relationships_csv)
        if self.resume:
            self.checkpoint_store.recover_interrupted_docs()
        self.checkpoint_store.start_run(self.run_id, self._run_args())

        await self.artifact_sink.ensure_collection(
            vector_size=self.dense_encoder.embedding_dimension,
            collection_name=self.collection_name,
        )

        try:
            with self.content_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if self.limit is not None and self.stats["docs_seen"] >= self.limit:
                        break
                    self.stats["docs_seen"] += 1
                    doc_id = str(row.get("id") or "").strip()
                    if not doc_id:
                        continue

                    status = self.checkpoint_store.get_doc_status(doc_id) if self.resume else None
                    if status == "done":
                        self.stats["docs_skipped"] += 1
                        continue

                    try:
                        self._chunk_document(doc_id, row)
                    except Exception as exc:
                        self.stats["docs_failed"] += 1
                        self.checkpoint_store.mark_failed(doc_id, self.run_id, str(exc))
                        self._write_dead_letter(doc_id, row, exc)
                        print(f"[failed:chunk] doc_id={doc_id} error={exc}", file=sys.stderr)
                        continue

                    # Flush when the buffer is full enough
                    if len(self._pending) >= self.gpu_batch_size:
                        await self._flush()

            # Final flush for remaining chunks
            if self._pending:
                await self._flush()

        except Exception:
            self.checkpoint_store.finish_run(self.run_id, "failed")
            raise

        self.checkpoint_store.finish_run(self.run_id, "completed")
        return self.stats

    # ----- phase 1: chunking (CPU) -----

    def _chunk_document(self, doc_id: str, row: dict[str, Any]) -> None:
        """Parse one document and append its chunks to the pending buffer."""
        self.checkpoint_store.mark_processing(doc_id, self.run_id)
        metadata = self.checkpoint_store.get_metadata(doc_id)
        relationships = self.checkpoint_store.get_relationships(doc_id)
        document_tree = build_document_tree(
            doc_id=doc_id,
            markdown=html_to_markdown(row.get("content_html")),
            metadata=metadata,
            relationships=relationships,
        )
        chunks = self.extractor.extract_document_chunks(document_tree.model_dump(mode="json"))

        if not chunks:
            # Document produced zero chunks – mark done immediately
            self.checkpoint_store.mark_done(doc_id, self.run_id, 0)
            self.stats["docs_empty"] += 1
            self.stats["docs_processed"] += 1
            print(f"[ok] doc_id={doc_id} chunks=0 (empty)")
            return

        # Register a document batch tracker
        doc_batch = DocumentBatch(doc_id=doc_id, total_chunks=len(chunks))
        self._doc_batches[doc_id] = doc_batch

        for chunk in chunks:
            payload = build_chunk_payload(chunk, self.pipeline_version)
            self._pending.append(
                PendingChunk(
                    doc_id=doc_id,
                    payload=payload,
                    context_text=payload["context_text"],
                )
            )

        self.stats["chunks_total"] += len(chunks)

    # ----- phase 2: GPU flush -----

    async def _flush(self) -> None:
        """Encode the current pending buffer, build points, write to sink."""
        if not self._pending:
            return

        batch = list(self._pending)
        self._pending.clear()

        texts = [item.context_text for item in batch]

        # ---- GPU embedding ----
        t0 = time.perf_counter()
        dense_vectors = self.dense_encoder.encode_documents(texts)
        embed_elapsed = time.perf_counter() - t0
        self.stats["total_embed_time_s"] += embed_elapsed
        self.stats["gpu_flushes"] += 1

        throughput = len(texts) / embed_elapsed if embed_elapsed > 0 else 0.0
        print(
            f"[gpu-flush #{self.stats['gpu_flushes']}] "
            f"chunks={len(texts)}  "
            f"time={embed_elapsed:.2f}s  "
            f"throughput={throughput:.0f} chunks/s"
        )

        # ---- Build PointStructs ----
        points: list[models.PointStruct] = []
        for index, item in enumerate(batch):
            point = models.PointStruct(
                id=deterministic_point_id(item.payload["chunk_id"], self.pipeline_version),
                vector={
                    DENSE_VECTOR_NAME: dense_vectors[index],
                    SPARSE_VECTOR_NAME: build_bm25_document(
                        item.context_text,
                        model_name=self.sparse_model_name,
                        options=self.bm25_options,
                    ),
                },
                payload=item.payload,
            )
            points.append(point)

        # ---- Write to sink in sub-batches ----
        for start in range(0, len(points), self.shard_size):
            sub_batch = points[start : start + self.shard_size]
            await self.artifact_sink.write_points(sub_batch, collection_name=self.collection_name)

        self.stats["chunks_upserted"] += len(points)

        # ---- Per-document completion tracking ----
        # Count how many chunks were flushed per document in this batch
        flushed_per_doc: dict[str, int] = {}
        for item in batch:
            flushed_per_doc[item.doc_id] = flushed_per_doc.get(item.doc_id, 0) + 1

        for doc_id, count in flushed_per_doc.items():
            doc_batch = self._doc_batches.get(doc_id)
            if doc_batch is None:
                continue
            doc_batch.flushed_chunks += count
            if doc_batch.is_complete:
                self.checkpoint_store.mark_done(doc_id, self.run_id, doc_batch.total_chunks)
                self.stats["docs_processed"] += 1
                print(f"[ok] doc_id={doc_id} chunks={doc_batch.total_chunks}")
                del self._doc_batches[doc_id]

    # ----- helpers -----

    def _write_dead_letter(self, doc_id: str, row: dict[str, Any], exc: Exception) -> None:
        self.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "doc_id": doc_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "row": {"id": row.get("id")},
        }
        with self.dead_letter_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _run_args(self) -> dict[str, Any]:
        return {
            "content_csv": str(self.content_csv),
            "metadata_csv": str(self.metadata_csv),
            "relationships_csv": str(self.relationships_csv),
            "collection_name": self.collection_name,
            "pipeline_version": self.pipeline_version,
            "gpu_batch_size": self.gpu_batch_size,
            "dense_model_name": self.dense_encoder.model_name,
            "device": self.dense_encoder.device,
            "embed_batch_size": self.dense_encoder.batch_size,
            "resume": self.resume,
            "limit": self.limit,
        }

    @staticmethod
    def _configure_csv_limits() -> None:
        max_size = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_size)
                return
            except OverflowError:
                max_size = max_size // 10


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()

    checkpoint_store = SQLiteCheckpointStore(args.checkpoint_db)
    dense_encoder = SentenceTransformerDenseEncoder(
        model_name=args.dense_model_name,
        device=args.device,
        batch_size=args.embed_batch_size,
    )
    artifact_sink = ArtifactPointSink(
        args.output_dir,
        dense_model_name=args.dense_model_name,
        pipeline_version=args.pipeline_version,
        sparse_model_name=args.sparse_model_name,
        bm25_options={
            "k": args.bm25_k,
            "b": args.bm25_b,
            "language": args.bm25_language,
        },
        shard_size=args.shard_size,
        append_existing=args.resume,
    )

    pipeline = BatchedHybridExportPipeline(
        checkpoint_store=checkpoint_store,
        dense_encoder=dense_encoder,
        artifact_sink=artifact_sink,
        content_csv=Path(args.content_csv),
        metadata_csv=Path(args.metadata_csv),
        relationships_csv=Path(args.relationships_csv),
        dead_letter_path=Path(args.dead_letter),
        collection_name=args.collection,
        pipeline_version=args.pipeline_version,
        sparse_model_name=args.sparse_model_name,
        bm25_options={
            "k": args.bm25_k,
            "b": args.bm25_b,
            "language": args.bm25_language,
        },
        gpu_batch_size=args.gpu_batch_size,
        shard_size=args.shard_size,
        resume=args.resume,
        limit=args.limit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        table_chunk_size=args.table_chunk_size,
        import_sidecar=not args.skip_sidecar_import,
    )

    try:
        stats = await pipeline.run()
        artifact_sink.close(stats)
    finally:
        checkpoint_store.close()

    print()
    print("=" * 60)
    print("Batched hybrid artifact export complete")
    print(f"output_dir : {args.output_dir}")
    print(f"gpu_batch  : {args.gpu_batch_size}")
    print("-" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
