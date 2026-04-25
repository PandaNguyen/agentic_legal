from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from app.core.config import Settings
from app.schemas.search import SearchFilters, SearchHit
from app.services.retrieval.checkpoint_store import SQLiteCheckpointStore
from app.services.retrieval.hybrid_support import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    SentenceTransformerDenseEncoder,
    build_qdrant_filter,
    build_sparse_vector,
    payload_index_specs,
)


class QdrantService:
    def __init__(
        self,
        settings: Settings,
        _openai_service: Any | None = None,
        *,
        client: AsyncQdrantClient | None = None,
        dense_encoder: SentenceTransformerDenseEncoder | None = None,
        checkpoint_store: SQLiteCheckpointStore | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.collection_name = settings.qdrant_collection_hybrid or settings.qdrant_collection
        self._dense_encoder = dense_encoder
        self._checkpoint_store = checkpoint_store

    async def ensure_collection(self) -> None:
        dense_encoder = self._get_dense_encoder()
        await self.ensure_hybrid_collection(vector_size=dense_encoder.embedding_dimension)

    async def ensure_hybrid_collection(self, vector_size: int, collection_name: str | None = None) -> None:
        target_collection = collection_name or self.collection_name
        if await self.client.collection_exists(target_collection):
            await self._validate_collection(target_collection, vector_size)
        else:
            await self.client.create_collection(
                collection_name=target_collection,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                        datatype=models.Datatype.FLOAT16,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                on_disk_payload=True,
            )

        for field_name, field_schema in payload_index_specs().items():
            await self.client.create_payload_index(
                collection_name=target_collection,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    async def upsert_points(
        self,
        points: list[models.PointStruct],
        collection_name: str | None = None,
    ) -> None:
        if not points:
            return
        await self.client.upsert(
            collection_name=collection_name or self.collection_name,
            points=points,
            wait=True,
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        search_mode: str = "hybrid",
        candidate_limit: int = 50,
        filters: SearchFilters | None = None,
        sector: str | None = None,
    ) -> list[SearchHit]:
        if filters is None and sector:
            filters = SearchFilters(nganh=[sector])

        mode = (search_mode or "hybrid").lower()
        limit = max(top_k, 1)
        candidate_limit = max(candidate_limit, limit)
        qfilter = build_qdrant_filter(filters)
        dense_query = self._get_dense_encoder().encode_query(query)

        sparse_query = None
        checkpoint_store = self._get_checkpoint_store(optional=True)
        if checkpoint_store is not None:
            sparse_query = build_sparse_vector(query, checkpoint_store, create_missing=False)

        if mode == "sparse" and sparse_query is None:
            raise ValueError("Sparse search requested but checkpoint vocabulary is unavailable or query has no known sparse terms")

        if mode == "hybrid" and sparse_query is not None:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using=DENSE_VECTOR_NAME,
                        limit=candidate_limit,
                        filter=qfilter,
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using=SPARSE_VECTOR_NAME,
                        limit=candidate_limit,
                        filter=qfilter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        elif mode == "sparse":
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_query,
                using=SPARSE_VECTOR_NAME,
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
            )
        else:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=dense_query,
                using=DENSE_VECTOR_NAME,
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
            )

        results: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchHit(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    doc_id=str(payload.get("doc_id", point.id)),
                    title=str(payload.get("title") or payload.get("doc_title") or "Unknown document"),
                    article_number=_clean_optional(payload.get("article_number") or payload.get("article")),
                    clause_number=_clean_optional(payload.get("clause_number") or payload.get("clause")),
                    point_number=_clean_optional(payload.get("point_number")),
                    source_node_type=_clean_optional(payload.get("source_node_type")),
                    so_ky_hieu=_clean_optional(payload.get("so_ky_hieu")),
                    loai_van_ban=_clean_optional(payload.get("loai_van_ban")),
                    co_quan_ban_hanh=_clean_optional(payload.get("co_quan_ban_hanh")),
                    pham_vi=_clean_optional(payload.get("pham_vi")),
                    tinh_trang_hieu_luc=_clean_optional(payload.get("tinh_trang_hieu_luc")),
                    text=str(payload.get("text", "")),
                    context_text=_clean_optional(payload.get("context_text")),
                    score=float(point.score or 0.0),
                    metadata={
                        "source_url": _clean_optional(payload.get("source_url")),
                        "nganh": _clean_optional(payload.get("nganh")),
                        "linh_vuc": _clean_optional(payload.get("linh_vuc")),
                        "pipeline_version": _clean_optional(payload.get("pipeline_version")),
                    },
                )
            )
        return results

    async def _validate_collection(self, collection_name: str, vector_size: int) -> None:
        collection = await self.client.get_collection(collection_name)
        params = collection.config.params
        vectors = params.vectors
        sparse_vectors = params.sparse_vectors or {}

        if not isinstance(vectors, dict) or DENSE_VECTOR_NAME not in vectors:
            raise ValueError(f"Collection {collection_name} does not expose named dense vector '{DENSE_VECTOR_NAME}'")
        dense_params = vectors[DENSE_VECTOR_NAME]
        if int(dense_params.size) != int(vector_size):
            raise ValueError(
                f"Collection {collection_name} dense vector size mismatch: expected {vector_size}, found {dense_params.size}"
            )
        if SPARSE_VECTOR_NAME not in sparse_vectors:
            raise ValueError(f"Collection {collection_name} does not expose sparse vector '{SPARSE_VECTOR_NAME}'")

    def _get_dense_encoder(self) -> SentenceTransformerDenseEncoder:
        if self._dense_encoder is None:
            self._dense_encoder = SentenceTransformerDenseEncoder(
                model_name=self.settings.dense_embedding_model,
                device=self.settings.dense_embedding_device,
                batch_size=self.settings.dense_embedding_batch_size,
            )
        return self._dense_encoder

    def _get_checkpoint_store(self, optional: bool = False) -> SQLiteCheckpointStore | None:
        if self._checkpoint_store is not None:
            return self._checkpoint_store

        checkpoint_path = Path(self.settings.ingest_checkpoint_db)
        if not checkpoint_path.exists():
            if optional:
                return None
            raise FileNotFoundError(
                f"Checkpoint database not found at {checkpoint_path}. Hybrid sparse search requires the ingest sidecar."
            )

        self._checkpoint_store = SQLiteCheckpointStore(checkpoint_path)
        self._checkpoint_store.init_schema()
        return self._checkpoint_store


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
