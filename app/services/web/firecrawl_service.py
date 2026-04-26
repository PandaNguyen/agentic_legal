from __future__ import annotations

import httpx

from app.core.config import Settings
from app.schemas.agentic import WebSearchHit


class FirecrawlSearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = "https://api.firecrawl.dev/v2"

    @property
    def is_available(self) -> bool:
        return bool(self.settings.enable_web_search and self.settings.firecrawl_api_key)

    async def search(
        self,
        *,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
    ) -> list[WebSearchHit]:
        if not self.is_available:
            raise RuntimeError("Web search is disabled or FIRECRAWL_API_KEY is missing")

        final_query = self._apply_domain_filter(query, allowed_domains)
        payload = {
            "query": final_query,
            "limit": max(1, min(max_results, 10)),
            "sources": ["web"],
            "country": "VN",
            "ignoreInvalidURLs": True,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.settings.firecrawl_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.base_url}/search", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        raw_results = data.get("data", {}).get("web", [])
        return [
            WebSearchHit(
                title=str(item.get("title") or item.get("url") or "Untitled"),
                url=str(item.get("url") or ""),
                description=item.get("description"),
                markdown=item.get("markdown"),
                metadata={"category": item.get("category")},
            )
            for item in raw_results
            if item.get("url")
        ]

    def _apply_domain_filter(self, query: str, allowed_domains: list[str] | None) -> str:
        domains = [domain.strip() for domain in allowed_domains or [] if domain.strip()]
        if not domains:
            return query
        domain_clause = " OR ".join(f"site:{domain}" for domain in domains)
        return f"({domain_clause}) {query}"

