from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from qdrant_client import models

from app.services.retrieval.hybrid_support import (
    BM25_MODEL_NAME,
    DEFAULT_BM25_OPTIONS,
    DENSE_VECTOR_NAME,
)

ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_NAME = "manifest.json"


@dataclass(slots=True)
class ArtifactShard:
    name: str
    point_count: int = 0


class ArtifactPointSink:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        dense_model_name: str,
        pipeline_version: str,
        sparse_model_name: str = BM25_MODEL_NAME,
        bm25_options: dict[str, Any] | None = None,
        shard_size: int = 10_000,
        manifest_name: str = DEFAULT_MANIFEST_NAME,
        append_existing: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dense_model_name = dense_model_name
        self.pipeline_version = pipeline_version
        self.sparse_model_name = sparse_model_name
        self.bm25_options = bm25_options or DEFAULT_BM25_OPTIONS
        self.shard_size = max(1, shard_size)
        self.manifest_name = manifest_name
        self.append_existing = append_existing
        self.collection_name: str | None = None
        self.vector_size: int | None = None
        self.point_count = 0
        self.doc_count = 0
        self.shards: list[ArtifactShard] = []
        self._current_handle: TextIO | None = None
        self._current_shard: ArtifactShard | None = None

    async def ensure_collection(self, vector_size: int, collection_name: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_output_dir()
        self.vector_size = int(vector_size)
        self.collection_name = collection_name

    async def write_points(self, points: list[models.PointStruct], collection_name: str) -> None:
        if self.collection_name is None:
            self.collection_name = collection_name
        for point in points:
            self._write_record(point)

    def close(self, stats: dict[str, Any] | None = None) -> None:
        self._close_current_shard()
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "pipeline_version": self.pipeline_version,
            "dense_model_name": self.dense_model_name,
            "sparse_model_name": self.sparse_model_name,
            "bm25_options": self.bm25_options,
            "collection_name": self.collection_name,
            "vector_size": self.vector_size,
            "point_count": self.point_count,
            "doc_count": int((stats or {}).get("docs_processed", 0)),
            "shards": [
                {"name": shard.name, "point_count": shard.point_count}
                for shard in self.shards
            ],
            "stats": stats or {},
        }
        manifest_path = self.output_dir / self.manifest_name
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _write_record(self, point: models.PointStruct) -> None:
        vector = point.vector
        if not isinstance(vector, dict):
            raise ValueError("Artifact export expects named vector points")

        dense = vector.get(DENSE_VECTOR_NAME)
        if dense is None:
            raise ValueError(f"Point {point.id} does not contain dense vector '{DENSE_VECTOR_NAME}'")

        payload = point.payload or {}
        if not payload.get("context_text"):
            raise ValueError(f"Point {point.id} payload must include context_text for local BM25 import")

        if self._current_handle is None or self._current_shard is None:
            self._open_next_shard()
        elif self._current_shard.point_count >= self.shard_size:
            self._close_current_shard()
            self._open_next_shard()

        assert self._current_handle is not None
        assert self._current_shard is not None
        record = {
            "id": str(point.id),
            "dense": dense,
            "payload": payload,
        }
        self._current_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._current_shard.point_count += 1
        self.point_count += 1

    def _open_next_shard(self) -> None:
        shard = ArtifactShard(name=f"points-{len(self.shards) + 1:06d}.jsonl.gz")
        self.shards.append(shard)
        self._current_shard = shard
        self._current_handle = gzip.open(self.output_dir / shard.name, "wt", encoding="utf-8")

    def _close_current_shard(self) -> None:
        if self._current_handle is not None:
            self._current_handle.close()
        self._current_handle = None
        self._current_shard = None

    def _prepare_output_dir(self) -> None:
        existing_shards = sorted(self.output_dir.glob("points-*.jsonl.gz"))
        manifest_path = self.output_dir / self.manifest_name
        if not self.append_existing:
            if existing_shards or manifest_path.exists():
                raise ValueError(
                    f"Artifact output directory {self.output_dir} already contains artifacts; "
                    "choose an empty directory or rerun with resume/append enabled"
                )
            return

        for shard_path in existing_shards:
            point_count = self._count_existing_shard_points(shard_path)
            self.shards.append(ArtifactShard(name=shard_path.name, point_count=point_count))
            self.point_count += point_count

    @staticmethod
    def _count_existing_shard_points(shard_path: Path) -> int:
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
