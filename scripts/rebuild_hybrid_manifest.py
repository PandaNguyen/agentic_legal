from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings  # noqa: E402
from app.services.retrieval.artifact_store import ARTIFACT_SCHEMA_VERSION, DEFAULT_MANIFEST_NAME  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Rebuild manifest.json from existing hybrid artifact shards.")
    parser.add_argument("--artifact-dir", default="data/artifacts/hybrid_qdrant", help="Directory containing points-*.jsonl.gz.")
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME, help="Manifest file name to write.")
    parser.add_argument("--dense-model-name", default=settings.dense_embedding_model, help="Dense embedding model used to create shards.")
    parser.add_argument("--vector-size", type=int, required=True, help="Dense vector dimension. mainguyen9/vietlegal-harrier-0.6b is 1024.")
    parser.add_argument("--collection", default=settings.qdrant_collection_hybrid, help="Target Qdrant collection name.")
    parser.add_argument("--sparse-model-name", default=settings.sparse_embedding_model, help="Sparse model name.")
    parser.add_argument("--bm25-k", type=float, default=settings.bm25_k, help="BM25 k parameter.")
    parser.add_argument("--bm25-b", type=float, default=settings.bm25_b, help="BM25 b parameter.")
    parser.add_argument("--bm25-language", default=settings.bm25_language, help="BM25 language.")
    parser.add_argument("--pipeline-version", default=settings.ingest_pipeline_version, help="Pipeline version used for deterministic IDs.")
    parser.add_argument("--checkpoint-db", default=None, help="Optional checkpoint SQLite file for doc/status stats.")
    parser.add_argument("--skip-corrupt-shards", action="store_true", help="Skip gzip shards truncated by an interrupted session.")
    parser.add_argument("--include-partial-corrupt-shards", action="store_true", help="Dangerous: include readable records from corrupt gzip shards in the manifest.")
    return parser


def count_shard_points(shard_path: Path) -> tuple[int, str | None]:
    count = 0
    try:
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except (EOFError, OSError, UnicodeDecodeError) as exc:
        return count, str(exc)
    return count, None


def load_checkpoint_stats(checkpoint_db: str | None) -> dict[str, Any]:
    if not checkpoint_db:
        return {}
    db_path = Path(checkpoint_db)
    if not db_path.exists():
        return {"checkpoint_db": str(db_path), "checkpoint_missing": True}

    conn = sqlite3.connect(db_path)
    try:
        status_counts = {
            str(status): int(count)
            for status, count in conn.execute(
                """
                SELECT status, COUNT(*)
                FROM doc_status
                GROUP BY status
                ORDER BY status
                """
            )
        }
        done_docs, chunks_done = conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(chunk_count), 0)
            FROM doc_status
            WHERE status = 'done'
            """
        ).fetchone()
    finally:
        conn.close()

    return {
        "checkpoint_db": str(db_path),
        "doc_status": status_counts,
        "done_docs": int(done_docs),
        "chunks_done": int(chunks_done),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    shard_paths = sorted(artifact_dir.glob("points-*.jsonl.gz"))
    if not shard_paths:
        raise SystemExit(f"No points-*.jsonl.gz shards found in {artifact_dir}")

    shards = []
    corrupt_shards = []
    point_count = 0
    for index, shard_path in enumerate(shard_paths, start=1):
        print(f"[{index}/{len(shard_paths)}] scanning {shard_path.name}...", flush=True)
        count, error = count_shard_points(shard_path)
        if error:
            corrupt_shards.append({"name": shard_path.name, "readable_point_count": count, "error": error})
            print(f"  CORRUPT readable_points={count} error={error}", flush=True)
            if args.include_partial_corrupt_shards:
                shards.append({"name": shard_path.name, "point_count": count, "corrupt": True})
                point_count += count
                continue
            if args.skip_corrupt_shards:
                continue
            raise SystemExit(
                f"Corrupt shard detected: {shard_path}. "
                "Rerun with --skip-corrupt-shards or delete the corrupt shard and rebuild again."
            )
        shards.append({"name": shard_path.name, "point_count": count})
        point_count += count

    checkpoint_stats = load_checkpoint_stats(args.checkpoint_db)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "pipeline_version": args.pipeline_version,
        "dense_model_name": args.dense_model_name,
        "sparse_model_name": args.sparse_model_name,
        "bm25_options": {
            "k": args.bm25_k,
            "b": args.bm25_b,
            "language": args.bm25_language,
        },
        "collection_name": args.collection,
        "vector_size": args.vector_size,
        "point_count": point_count,
        "doc_count": int(checkpoint_stats.get("done_docs") or 0),
        "shards": shards,
        "stats": {
            "rebuild_manifest": True,
            "corrupt_shards": corrupt_shards,
            **checkpoint_stats,
        },
    }

    manifest_path = artifact_dir / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {manifest_path}")
    print(f"shards: {len(shards)}")
    print(f"points: {point_count}")
    if checkpoint_stats:
        print(f"done_docs: {checkpoint_stats.get('done_docs', 0)}")
        print(f"chunks_done: {checkpoint_stats.get('chunks_done', 0)}")


if __name__ == "__main__":
    main()
