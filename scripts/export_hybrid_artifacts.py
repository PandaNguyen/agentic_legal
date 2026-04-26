from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings  # noqa: E402
from app.services.retrieval.artifact_store import ArtifactPointSink  # noqa: E402
from app.services.retrieval.checkpoint_store import SQLiteCheckpointStore  # noqa: E402
from app.services.retrieval.hybrid_ingest import HybridIngestConfig, HybridIngestPipeline  # noqa: E402
from app.services.retrieval.hybrid_support import SentenceTransformerDenseEncoder  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Export dense-vector legal chunks as local-importable Qdrant artifacts.")
    parser.add_argument("--content-csv", default="data/content.csv", help="CSV containing id,content_html columns.")
    parser.add_argument("--metadata-csv", default="data/metadata.csv", help="CSV containing document metadata.")
    parser.add_argument("--relationships-csv", default="data/relationships.csv", help="CSV containing document relationships.")
    parser.add_argument("--output-dir", default="data/artifacts/hybrid_qdrant", help="Directory for manifest and shard files.")
    parser.add_argument("--dense-model-name", default=settings.dense_embedding_model, help="Hugging Face/SentenceTransformers model.")
    parser.add_argument("--sparse-model-name", default=settings.sparse_embedding_model, help="Sparse model name for local Qdrant BM25 import.")
    parser.add_argument("--bm25-k", type=float, default=settings.bm25_k, help="BM25 k parameter.")
    parser.add_argument("--bm25-b", type=float, default=settings.bm25_b, help="BM25 b parameter.")
    parser.add_argument("--bm25-language", default=settings.bm25_language, help="BM25 language; use none for Vietnamese legal text.")
    parser.add_argument("--collection", default=settings.qdrant_collection_hybrid, help="Target Qdrant collection name encoded in manifest.")
    parser.add_argument("--device", default=settings.dense_embedding_device, help="Embedding device, e.g. cuda or cpu.")
    parser.add_argument("--embed-batch-size", type=int, default=settings.dense_embedding_batch_size, help="Dense embedding batch size.")
    parser.add_argument("--shard-size", type=int, default=10_000, help="Number of points per artifact shard.")
    parser.add_argument("--checkpoint-db", default=settings.ingest_checkpoint_db, help="SQLite checkpoint/metadata sidecar path.")
    parser.add_argument("--dead-letter", default="data/processed/hybrid_artifact_dead_letter.jsonl", help="JSONL file for failed rows.")
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint DB and skip completed documents.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of CSV rows to scan for a dry run.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Approximate chunk size in tokens.")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Overlap for long text chunks.")
    parser.add_argument("--table-chunk-size", type=int, default=500, help="Approximate chunk size for tables.")
    parser.add_argument("--pipeline-version", default=settings.ingest_pipeline_version, help="Deterministic point-id version prefix.")
    parser.add_argument("--skip-sidecar-import", action="store_true", help="Skip importing metadata/relationships CSV into SQLite.")
    return parser


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
    pipeline = HybridIngestPipeline(
        HybridIngestConfig(
            content_csv=Path(args.content_csv),
            metadata_csv=Path(args.metadata_csv),
            relationships_csv=Path(args.relationships_csv),
            checkpoint_db=Path(args.checkpoint_db),
            dead_letter_path=Path(args.dead_letter),
            collection_name=args.collection,
            dense_model_name=args.dense_model_name,
            device=args.device,
            embed_batch_size=args.embed_batch_size,
            qdrant_batch_size=args.shard_size,
            resume=args.resume,
            limit=args.limit,
            pipeline_version=args.pipeline_version,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            table_chunk_size=args.table_chunk_size,
            import_sidecar=not args.skip_sidecar_import,
            sparse_model_name=args.sparse_model_name,
            bm25_options={
                "k": args.bm25_k,
                "b": args.bm25_b,
                "language": args.bm25_language,
            },
        ),
        checkpoint_store=checkpoint_store,
        point_sink=artifact_sink,
        dense_encoder=dense_encoder,
    )

    try:
        stats = await pipeline.run()
        artifact_sink.close(stats)
    finally:
        checkpoint_store.close()

    print("Hybrid artifact export complete")
    print(f"output_dir: {args.output_dir}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
