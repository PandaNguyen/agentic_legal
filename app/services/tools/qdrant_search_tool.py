from __future__ import annotations

from app.schemas.agentic import ToolResult
from app.schemas.search import SearchFilters
from app.services.retrieval.qdrant_service import QdrantService


class QdrantSearchTool:
    name = "qdrant_search"
    description = "Search the local ingested Vietnam legal corpus with hybrid dense/BM25 retrieval."

    def __init__(self, qdrant_service: QdrantService) -> None:
        self.qdrant_service = qdrant_service

    async def run(
        self,
        *,
        query: str,
        search_mode: str = "hybrid",
        top_k: int = 8,
        candidate_limit: int = 200,
        filters: SearchFilters | None = None,
    ) -> ToolResult:
        try:
            hits = await self.qdrant_service.search(
                query=query,
                top_k=top_k,
                search_mode=search_mode,
                candidate_limit=candidate_limit,
                filters=filters,
            )
            return ToolResult(tool_name="qdrant_search", query=query, qdrant_hits=hits)
        except Exception as exc:
            return ToolResult(
                tool_name="qdrant_search",
                query=query,
                available=False,
                error=str(exc),
            )
