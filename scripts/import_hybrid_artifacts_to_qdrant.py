from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings  # noqa: E402
from app.services.retrieval.artifact_import import import_artifacts_to_qdrant, load_manifest  # noqa: E402
from app.services.retrieval.qdrant_service import QdrantService  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Import Kaggle-exported hybrid artifacts into local Qdrant.")
    parser.add_argument("--artifact-dir", default="data/artifacts/hybrid_qdrant", help="Directory containing manifest.json and point shards.")
    parser.add_argument("--qdrant-url", default=settings.qdrant_url, help="Local Qdrant URL, e.g. http://localhost:6333.")
    parser.add_argument("--qdrant-api-key", default=settings.qdrant_api_key, help="Optional Qdrant API key.")
    parser.add_argument("--collection", default=None, help="Override target collection name from manifest.")
    parser.add_argument("--batch-size", type=int, default=128, help="Qdrant upsert batch size.")
    parser.add_argument("--skip-version-check", action="store_true", help="Skip Qdrant >=1.15.3 native BM25 validation.")
    return parser


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()
    manifest = load_manifest(args.artifact_dir)
    settings = get_settings().model_copy(
        update={
            "qdrant_url": args.qdrant_url,
            "qdrant_api_key": args.qdrant_api_key,
            "qdrant_collection_hybrid": args.collection or manifest["collection_name"],
            "dense_embedding_model": manifest["dense_model_name"],
            "sparse_embedding_model": manifest["sparse_model_name"],
            "bm25_k": float(manifest["bm25_options"].get("k", 1.2)),
            "bm25_b": float(manifest["bm25_options"].get("b", 0.75)),
            "bm25_language": str(manifest["bm25_options"].get("language", "none")),
        }
    )
    qdrant_service = QdrantService(settings)
    stats = await import_artifacts_to_qdrant(
        args.artifact_dir,
        qdrant_service,
        collection_name=args.collection,
        batch_size=args.batch_size,
        validate_bm25=not args.skip_version_check,
    )

    print("Hybrid artifact import complete")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
