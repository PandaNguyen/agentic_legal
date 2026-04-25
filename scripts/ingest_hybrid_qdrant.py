from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings  # noqa: E402
from app.services.retrieval.checkpoint_store import SQLiteCheckpointStore  # noqa: E402
from app.services.retrieval.hybrid_ingest import HybridIngestConfig, HybridIngestPipeline  # noqa: E402
from app.services.retrieval.hybrid_support import SentenceTransformerDenseEncoder  # noqa: E402
from app.services.retrieval.qdrant_service import QdrantService  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Stream large legal CSV files into a hybrid dense+sparse Qdrant collection.")
    parser.add_argument("--content-csv", default="data/content.csv", help="CSV containing id,content_html columns.")
    parser.add_argument("--metadata-csv", default="data/metadata.csv", help="CSV containing document metadata.")
    parser.add_argument("--relationships-csv", default="data/relationships.csv", help="CSV containing document relationships.")
    parser.add_argument("--dense-model-name", default=settings.dense_embedding_model, help="Hugging Face/SentenceTransformers model.")
    parser.add_argument("--collection", default=settings.qdrant_collection_hybrid, help="Target Qdrant collection name.")
    parser.add_argument("--device", default=settings.dense_embedding_device, help="Embedding device, e.g. cuda or cpu.")
    parser.add_argument("--embed-batch-size", type=int, default=settings.dense_embedding_batch_size, help="Dense embedding batch size.")
    parser.add_argument("--qdrant-batch-size", type=int, default=64, help="Qdrant upsert batch size.")
    parser.add_argument("--checkpoint-db", default=settings.ingest_checkpoint_db, help="SQLite checkpoint/vocab sidecar path.")
    parser.add_argument("--dead-letter", default="data/processed/hybrid_dead_letter.jsonl", help="JSONL file for failed rows.")
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
    settings = get_settings().model_copy(
        update={
            "qdrant_collection_hybrid": args.collection,
            "dense_embedding_model": args.dense_model_name,
            "dense_embedding_device": args.device,
            "dense_embedding_batch_size": args.embed_batch_size,
            "ingest_checkpoint_db": args.checkpoint_db,
            "ingest_pipeline_version": args.pipeline_version,
        }
    )

    checkpoint_store = SQLiteCheckpointStore(args.checkpoint_db)
    dense_encoder = SentenceTransformerDenseEncoder(
        model_name=args.dense_model_name,
        device=args.device,
        batch_size=args.embed_batch_size,
    )
    qdrant_service = QdrantService(
        settings,
        client=None,
        dense_encoder=dense_encoder,
        checkpoint_store=checkpoint_store,
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
            qdrant_batch_size=args.qdrant_batch_size,
            resume=args.resume,
            limit=args.limit,
            pipeline_version=args.pipeline_version,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            table_chunk_size=args.table_chunk_size,
            import_sidecar=not args.skip_sidecar_import,
        ),
        qdrant_service=qdrant_service,
        checkpoint_store=checkpoint_store,
        dense_encoder=dense_encoder,
    )

    try:
        stats = await pipeline.run()
    finally:
        checkpoint_store.close()

    print("Hybrid ingest complete")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
