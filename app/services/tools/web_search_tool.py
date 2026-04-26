from __future__ import annotations

from app.schemas.agentic import ToolResult
from app.services.web.firecrawl_service import FirecrawlSearchService


DEFAULT_LEGAL_DOMAINS = [
    "vbpl.vn",
    "thuvienphapluat.vn",
    "vanban.chinhphu.vn",
    "congbobanan.toaan.gov.vn",
    "toaan.gov.vn",
]


class WebSearchTool:
    name = "web_search"
    description = "Search trusted Vietnam legal web sources through Firecrawl when enabled."

    def __init__(self, firecrawl_service: FirecrawlSearchService) -> None:
        self.firecrawl_service = firecrawl_service

    async def run(
        self,
        *,
        query: str,
        allowed_domains: list[str] | None = None,
        max_results: int = 5,
    ) -> ToolResult:
        if not self.firecrawl_service.is_available:
            return ToolResult(
                tool_name="web_search",
                query=query,
                available=False,
                error="Web search is disabled or FIRECRAWL_API_KEY is missing",
            )

        try:
            hits = await self.firecrawl_service.search(
                query=query,
                max_results=max_results,
                allowed_domains=allowed_domains or DEFAULT_LEGAL_DOMAINS,
            )
            return ToolResult(tool_name="web_search", query=query, web_hits=hits)
        except Exception as exc:
            return ToolResult(
                tool_name="web_search",
                query=query,
                available=False,
                error=str(exc),
            )

