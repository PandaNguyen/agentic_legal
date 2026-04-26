from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import models

from app.services.retrieval.artifact_store import ARTIFACT_SCHEMA_VERSION, DEFAULT_MANIFEST_NAME
from app.services.retrieval.hybrid_support import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, build_bm25_document
from app.services.retrieval.qdrant_service import QdrantService


def load_manifest(artifact_dir: str | Path, manifest_name: str = DEFAULT_MANIFEST_NAME) -> dict[str, Any]:
    manifest_path = Path(artifact_dir) / manifest_name
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", 0)) != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema_version={manifest.get('schema_version')}; expected {ARTIFACT_SCHEMA_VERSION}"
        )
    return manifest


def iter_artifact_points(
    artifact_dir: str | Path,
    manifest: dict[str, Any],
) -> Iterable[models.PointStruct]:
    artifact_path = Path(artifact_dir)
    sparse_model_name = str(manifest.get("sparse_model_name") or "Qdrant/bm25")
    bm25_options = dict(manifest.get("bm25_options") or {})

    for shard in manifest.get("shards", []):
        shard_name = str(shard["name"])
        shard_path = artifact_path / shard_name
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                payload = dict(record.get("payload") or {})
                context_text = payload.get("context_text")
                if not context_text:
                    raise ValueError(f"Artifact point {record.get('id')} is missing payload.context_text")
                yield models.PointStruct(
                    id=record["id"],
                    vector={
                        DENSE_VECTOR_NAME: record["dense"],
                        SPARSE_VECTOR_NAME: build_bm25_document(
                            str(context_text),
                            model_name=sparse_model_name,
                            options=bm25_options,
                        ),
                    },
                    payload=payload,
                )


async def import_artifacts_to_qdrant(
    artifact_dir: str | Path,
    qdrant_service: QdrantService,
    *,
    collection_name: str | None = None,
    batch_size: int = 128,
    validate_bm25: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(artifact_dir)
    target_collection = collection_name or str(manifest["collection_name"])
    vector_size = int(manifest["vector_size"])
    if validate_bm25:
        await qdrant_service.validate_native_bm25_support()
    await qdrant_service.ensure_hybrid_collection(vector_size=vector_size, collection_name=target_collection)

    imported_points = 0
    batch: list[models.PointStruct] = []
    for point in iter_artifact_points(artifact_dir, manifest):
        batch.append(point)
        if len(batch) >= batch_size:
            await qdrant_service.upsert_points(batch, collection_name=target_collection)
            imported_points += len(batch)
            batch.clear()

    if batch:
        await qdrant_service.upsert_points(batch, collection_name=target_collection)
        imported_points += len(batch)

    return {
        "collection_name": target_collection,
        "vector_size": vector_size,
        "imported_points": imported_points,
        "manifest_points": int(manifest.get("point_count") or 0),
        "shard_count": len(manifest.get("shards", [])),
    }
