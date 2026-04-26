from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import models  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.schemas.search import SearchFilters  # noqa: E402
from app.services.retrieval.hybrid_support import (  # noqa: E402
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    SentenceTransformerDenseEncoder,
    build_bm25_document,
    deterministic_point_id,
)
from app.services.retrieval.qdrant_service import QdrantService  # noqa: E402


SAMPLE_CHUNKS = [
    {
        "chunk_id": "demo:chunk:1",
        "doc_id": "bo_luat_lao_dong_demo_1",
        "title": "Bộ luật Lao động - Demo",
        "article_number": "Điều 35",
        "clause_number": "Khoản 1",
        "source_node_type": "article",
        "text": "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước theo thời hạn luật định.",
        "context_text": "Bộ luật Lao động - Demo\nĐiều 35\nNgười lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước theo thời hạn luật định.",
        "nganh": "Lao động",
        "nganh_slug": "lao_dong",
        "pipeline_version": "hybrid_v1",
        "token_count": 42,
    },
    {
        "chunk_id": "demo:chunk:2",
        "doc_id": "bo_luat_lao_dong_demo_2",
        "title": "Bộ luật Lao động - Demo",
        "article_number": "Điều 40",
        "clause_number": None,
        "source_node_type": "article",
        "text": "Người lao động đơn phương chấm dứt hợp đồng trái pháp luật có thể phải bồi thường trong một số trường hợp.",
        "context_text": "Bộ luật Lao động - Demo\nĐiều 40\nNgười lao động đơn phương chấm dứt hợp đồng trái pháp luật có thể phải bồi thường trong một số trường hợp.",
        "nganh": "Lao động",
        "nganh_slug": "lao_dong",
        "pipeline_version": "hybrid_v1",
        "token_count": 39,
    },
]


async def main() -> None:
    settings = get_settings()

    dense_encoder = SentenceTransformerDenseEncoder(
        model_name=settings.dense_embedding_model,
        device=settings.dense_embedding_device,
        batch_size=settings.dense_embedding_batch_size,
    )
    qdrant_service = QdrantService(
        settings,
        dense_encoder=dense_encoder,
    )
    await qdrant_service.ensure_hybrid_collection(dense_encoder.embedding_dimension)

    vectors = dense_encoder.encode_documents([chunk["context_text"] for chunk in SAMPLE_CHUNKS])
    points: list[models.PointStruct] = []
    for index, chunk in enumerate(SAMPLE_CHUNKS):
        points.append(
            models.PointStruct(
                id=deterministic_point_id(chunk["chunk_id"], settings.ingest_pipeline_version),
                vector={
                    DENSE_VECTOR_NAME: vectors[index],
                    SPARSE_VECTOR_NAME: build_bm25_document(
                        chunk["context_text"],
                        model_name=settings.sparse_embedding_model,
                        options={
                            "k": settings.bm25_k,
                            "b": settings.bm25_b,
                            "language": settings.bm25_language,
                        },
                    ),
                },
                payload=chunk,
            )
        )

    await qdrant_service.upsert_points(points)
    results = await qdrant_service.search(
        query="đơn phương chấm dứt hợp đồng lao động",
        filters=SearchFilters(nganh=["lao_dong"]),
    )
    print(f"Seeded sample chunks to {settings.qdrant_collection_hybrid}; sample_hits={len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
