from __future__ import annotations

import csv
import json
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import models

from app.services.chunking.html_to_markdown import html_to_markdown
from app.services.chunking.legal_chunk_extractor import ChunkingConfig, LegalTreeChunkExtractor
from app.services.chunking.legal_tree_builder import build_document_tree
from app.services.retrieval.checkpoint_store import SQLiteCheckpointStore
from app.services.retrieval.hybrid_support import (
    DENSE_VECTOR_NAME,
    DEFAULT_PIPELINE_VERSION,
    SPARSE_VECTOR_NAME,
    BM25_MODEL_NAME,
    DEFAULT_BM25_OPTIONS,
    SentenceTransformerDenseEncoder,
    build_bm25_document,
    build_chunk_payload,
    deterministic_point_id,
)
from app.services.retrieval.qdrant_service import QdrantService


@dataclass(slots=True)
class HybridIngestConfig:
    content_csv: Path
    metadata_csv: Path
    relationships_csv: Path
    checkpoint_db: Path
    dead_letter_path: Path
    collection_name: str
    dense_model_name: str
    device: str | None = None
    embed_batch_size: int = 32
    qdrant_batch_size: int = 64
    resume: bool = False
    limit: int | None = None
    pipeline_version: str = DEFAULT_PIPELINE_VERSION
    chunk_size: int = 500
    chunk_overlap: int = 64
    table_chunk_size: int = 500
    import_sidecar: bool = True
    sparse_model_name: str = BM25_MODEL_NAME
    bm25_options: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_BM25_OPTIONS))


class HybridPointSink(Protocol):
    async def ensure_collection(self, vector_size: int, collection_name: str) -> None:
        ...

    async def write_points(self, points: list[models.PointStruct], collection_name: str) -> None:
        ...


class QdrantPointSink:
    def __init__(self, qdrant_service: QdrantService) -> None:
        self.qdrant_service = qdrant_service

    async def ensure_collection(self, vector_size: int, collection_name: str) -> None:
        await self.qdrant_service.ensure_hybrid_collection(
            vector_size=vector_size,
            collection_name=collection_name,
        )

    async def write_points(self, points: list[models.PointStruct], collection_name: str) -> None:
        await self.qdrant_service.upsert_points(points, collection_name=collection_name)


class HybridIngestPipeline:
    def __init__(
        self,
        config: HybridIngestConfig,
        checkpoint_store: SQLiteCheckpointStore,
        qdrant_service: QdrantService | None = None,
        point_sink: HybridPointSink | None = None,
        dense_encoder: SentenceTransformerDenseEncoder | None = None,
    ) -> None:
        self.config = config
        self.checkpoint_store = checkpoint_store
        if point_sink is None:
            if qdrant_service is None:
                raise ValueError("Either qdrant_service or point_sink must be provided")
            point_sink = QdrantPointSink(qdrant_service)
        self.point_sink = point_sink
        self.dense_encoder = dense_encoder or SentenceTransformerDenseEncoder(
            model_name=config.dense_model_name,
            device=config.device,
            batch_size=config.embed_batch_size,
        )
        self.extractor = LegalTreeChunkExtractor(
            ChunkingConfig(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                table_chunk_size=config.table_chunk_size,
                include_relationships=False,
            )
        )
        self.run_id = f"ing_{uuid.uuid4().hex[:12]}"

    async def run(self) -> dict[str, int]:
        self._configure_csv_limits()
        self.checkpoint_store.init_schema()
        if self.config.import_sidecar:
            self.checkpoint_store.import_metadata_csv(self.config.metadata_csv)
            self.checkpoint_store.import_relationships_csv(self.config.relationships_csv)
        if self.config.resume:
            self.checkpoint_store.recover_interrupted_docs()
        self.checkpoint_store.start_run(self.run_id, self._run_args())

        await self.point_sink.ensure_collection(
            vector_size=self.dense_encoder.embedding_dimension,
            collection_name=self.config.collection_name,
        )

        stats = {
            "docs_seen": 0,
            "docs_processed": 0,
            "docs_skipped": 0,
            "docs_failed": 0,
            "chunks_upserted": 0,
        }

        try:
            with self.config.content_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if self.config.limit is not None and stats["docs_seen"] >= self.config.limit:
                        break
                    stats["docs_seen"] += 1
                    doc_id = str(row.get("id") or "").strip()
                    if not doc_id:
                        continue

                    status = self.checkpoint_store.get_doc_status(doc_id) if self.config.resume else None
                    if status == "done":
                        stats["docs_skipped"] += 1
                        continue

                    try:
                        chunk_count = await self._process_row(doc_id, row)
                        stats["docs_processed"] += 1
                        stats["chunks_upserted"] += chunk_count
                        print(f"[ok] doc_id={doc_id} chunks={chunk_count}")
                    except Exception as exc:
                        stats["docs_failed"] += 1
                        self.checkpoint_store.mark_failed(doc_id, self.run_id, str(exc))
                        self._write_dead_letter(doc_id, row, exc)
                        print(f"[failed] doc_id={doc_id} error={exc}", file=sys.stderr)
        except Exception:
            self.checkpoint_store.finish_run(self.run_id, "failed")
            raise

        self.checkpoint_store.finish_run(self.run_id, "completed")
        return stats

    async def _process_row(self, doc_id: str, row: dict[str, Any]) -> int:
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
            self.checkpoint_store.mark_done(doc_id, self.run_id, 0)
            return 0

        payloads = [build_chunk_payload(chunk, self.config.pipeline_version) for chunk in chunks]
        dense_vectors = self.dense_encoder.encode_documents([payload["context_text"] for payload in payloads])

        points = [
            models.PointStruct(
                id=deterministic_point_id(payload["chunk_id"], self.config.pipeline_version),
                vector={
                    DENSE_VECTOR_NAME: dense_vectors[index],
                    SPARSE_VECTOR_NAME: build_bm25_document(
                        payload["context_text"],
                        model_name=self.config.sparse_model_name,
                        options=self.config.bm25_options,
                    ),
                },
                payload=payload,
            )
            for index, payload in enumerate(payloads)
        ]

        for start in range(0, len(points), self.config.qdrant_batch_size):
            batch = points[start : start + self.config.qdrant_batch_size]
            await self.point_sink.write_points(batch, collection_name=self.config.collection_name)

        self.checkpoint_store.mark_done(doc_id, self.run_id, len(points))
        return len(points)

    def _write_dead_letter(self, doc_id: str, row: dict[str, Any], exc: Exception) -> None:
        self.config.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "doc_id": doc_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "row": {"id": row.get("id")},
        }
        with self.config.dead_letter_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _run_args(self) -> dict[str, Any]:
        return {
            "content_csv": str(self.config.content_csv),
            "metadata_csv": str(self.config.metadata_csv),
            "relationships_csv": str(self.config.relationships_csv),
            "checkpoint_db": str(self.config.checkpoint_db),
            "collection_name": self.config.collection_name,
            "dense_model_name": self.config.dense_model_name,
            "device": self.config.device,
            "embed_batch_size": self.config.embed_batch_size,
            "qdrant_batch_size": self.config.qdrant_batch_size,
            "resume": self.config.resume,
            "limit": self.config.limit,
            "pipeline_version": self.config.pipeline_version,
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
