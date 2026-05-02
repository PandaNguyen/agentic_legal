from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from app.core.config import Settings
from app.schemas.search import SearchFilters, SearchHit
from app.services.retrieval.checkpoint_store import SQLiteCheckpointStore
from app.services.retrieval.centroids import compute_dynamic_centroids, max_cosine_similarity
from app.services.retrieval.filter_policy import sanitize_search_filters
from app.services.retrieval.hybrid_support import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    SentenceTransformerDenseEncoder,
    build_bm25_document,
    build_qdrant_filter,
    payload_index_specs,
)
from app.services.retrieval.local_reranker import LocalReranker

MIN_NATIVE_BM25_QDRANT_VERSION = (1, 15, 3)


class QdrantService:
    def __init__(
        self,
        settings: Settings,
        _openai_service: Any | None = None,
        *,
        client: AsyncQdrantClient | None = None,
        dense_encoder: SentenceTransformerDenseEncoder | None = None,
        checkpoint_store: SQLiteCheckpointStore | None = None,
        reranker: LocalReranker | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=_api_key_for_qdrant_url(settings.qdrant_url, settings.qdrant_api_key),
        )
        self.collection_name = settings.qdrant_collection_hybrid or settings.qdrant_collection
        self._dense_encoder = dense_encoder
        self._checkpoint_store = checkpoint_store
        self._reranker = reranker
        self.sparse_model_name = settings.sparse_embedding_model
        self.bm25_options = {
            "k": settings.bm25_k,
            "b": settings.bm25_b,
            "language": settings.bm25_language,
        }

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
        candidate_limit: int = 200,
        filters: SearchFilters | None = None,
        sector: str | None = None,
    ) -> list[SearchHit]:
        if filters is None and sector:
            filters = SearchFilters(nganh=[sector])
        filters = sanitize_search_filters(filters)

        mode = (search_mode or "hybrid").lower()
        limit = max(top_k, 1)
        candidate_limit = max(candidate_limit, limit, int(self.settings.retrieval_initial_candidate_limit))
        qfilter = build_qdrant_filter(filters)
        dense_query = self._get_dense_encoder().encode_query(query)
        sparse_query = self._build_sparse_query(query)

        if mode == "hybrid":
            return await self._search_hybrid_enhanced(
                query=query,
                dense_query=dense_query,
                sparse_query=sparse_query,
                qfilter=qfilter,
                filters=filters,
                top_k=limit,
                candidate_limit=candidate_limit,
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

        return [self._point_to_search_hit(point) for point in response.points]

    async def _search_hybrid_enhanced(
        self,
        *,
        query: str,
        dense_query: list[float],
        sparse_query: models.Document,
        qfilter: models.Filter | None,
        filters: SearchFilters | None,
        top_k: int,
        candidate_limit: int,
    ) -> list[SearchHit]:
        candidate_response = await self._query_hybrid(
            dense_query=dense_query,
            sparse_query=sparse_query,
            qfilter=qfilter,
            limit=candidate_limit,
            with_vectors=True,
        )
        candidate_points = list(candidate_response.points)
        if not candidate_points:
            return []
        if len(candidate_points) <= top_k:
            return [self._point_to_search_hit(point) for point in candidate_points[:top_k]]

        selected_doc_ids = self._select_document_ids(candidate_points, dense_query)
        if not selected_doc_ids:
            fallback_hits = [self._point_to_search_hit(point) for point in candidate_points]
            return self._rerank_or_slice(query, fallback_hits, top_k)

        chunk_filters = self._merge_doc_filter(filters, selected_doc_ids)
        chunk_qfilter = build_qdrant_filter(chunk_filters)
        chunk_limit = max(top_k, int(self.settings.retrieval_chunk_candidate_limit))
        chunk_response = await self._query_hybrid(
            dense_query=dense_query,
            sparse_query=sparse_query,
            qfilter=chunk_qfilter,
            limit=chunk_limit,
            with_vectors=False,
        )
        chunk_hits = [self._point_to_search_hit(point) for point in chunk_response.points]
        return self._rerank_or_slice(query, chunk_hits, top_k)

    async def _query_hybrid(
        self,
        *,
        dense_query: list[float],
        sparse_query: models.Document,
        qfilter: models.Filter | None,
        limit: int,
        with_vectors: bool,
    ):
        return await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR_NAME,
                    limit=limit,
                    filter=qfilter,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR_NAME,
                    limit=limit,
                    filter=qfilter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )

    def _select_document_ids(self, points: list[Any], dense_query: list[float]) -> list[str]:
        vectors_by_doc: dict[str, list[list[float]]] = {}
        for point in points:
            payload = point.payload or {}
            doc_id = _clean_optional(payload.get("doc_id"))
            vector = _extract_dense_vector(getattr(point, "vector", None))
            if doc_id and vector:
                vectors_by_doc.setdefault(doc_id, []).append(vector)

        scored_docs: list[tuple[str, float]] = []
        for doc_id, vectors in vectors_by_doc.items():
            centroids, _ = compute_dynamic_centroids(
                vectors,
                max_chunks_per_cluster=int(self.settings.retrieval_max_chunks_per_cluster),
            )
            score = max_cosine_similarity(dense_query, centroids)
            scored_docs.append((doc_id, score))

        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in scored_docs[: max(1, int(self.settings.retrieval_document_top_k))]]

    def _merge_doc_filter(self, filters: SearchFilters | None, doc_ids: list[str]) -> SearchFilters:
        data = filters.model_dump() if filters else {}
        data["doc_ids"] = doc_ids
        return SearchFilters.model_validate(data)

    def _rerank_or_slice(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        if self.settings.enable_local_rerank and len(hits) > top_k:
            return self._get_reranker().rerank(query, hits, top_k)
        return hits[:top_k]

    def _point_to_search_hit(self, point: Any) -> SearchHit:
        payload = point.payload or {}
        return SearchHit(
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

    async def validate_native_bm25_support(self) -> None:
        info = await self.client.info()
        version = _parse_version(info.version)
        if version < MIN_NATIVE_BM25_QDRANT_VERSION:
            required = ".".join(str(part) for part in MIN_NATIVE_BM25_QDRANT_VERSION)
            raise RuntimeError(
                f"Qdrant {required}+ is required for native BM25 sparse vectors; server reports {info.version}"
            )

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

    def _build_sparse_query(self, query: str) -> models.Document:
        return build_bm25_document(
            query,
            model_name=self.sparse_model_name,
            options=self.bm25_options,
        )

    def _get_dense_encoder(self) -> SentenceTransformerDenseEncoder:
        if self._dense_encoder is None:
            self._dense_encoder = SentenceTransformerDenseEncoder(
                model_name=self.settings.dense_embedding_model,
                device=self.settings.dense_embedding_device,
                batch_size=self.settings.dense_embedding_batch_size,
            )
        return self._dense_encoder

    def _get_reranker(self) -> LocalReranker:
        if self._reranker is None:
            self._reranker = LocalReranker(self.settings)
        return self._reranker

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


def _extract_dense_vector(vector: Any) -> list[float] | None:
    if vector is None:
        return None
    if isinstance(vector, dict):
        raw_vector = vector.get(DENSE_VECTOR_NAME)
    else:
        raw_vector = vector
    if raw_vector is None:
        return None
    if hasattr(raw_vector, "tolist"):
        raw_vector = raw_vector.tolist()
    if not isinstance(raw_vector, list):
        return None
    try:
        return [float(item) for item in raw_vector]
    except (TypeError, ValueError):
        return None


def _parse_version(version: str) -> tuple[int, int, int]:
    parts = []
    for raw_part in str(version).split(".")[:3]:
        digits = "".join(ch for ch in raw_part if ch.isdigit())
        parts.append(int(digits or "0"))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)  # type: ignore[return-value]


def _api_key_for_qdrant_url(url: str, api_key: str | None) -> str | None:
    if not api_key:
        return None
    host = urlparse(url).hostname
    if host in {"localhost", "127.0.0.1", "::1"}:
        return None
    return api_key
