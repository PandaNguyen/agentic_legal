from __future__ import annotations

from app.schemas.agentic import EvidenceItem, EvidencePacket, ToolResult


class EvidenceAssemblerService:
    def assemble(self, tool_results: list[ToolResult]) -> EvidencePacket:
        items: list[EvidenceItem] = []
        seen: set[str] = set()
        qdrant_index = 1
        web_index = 1
        notes: list[str] = []

        for result in tool_results:
            if not result.available:
                notes.append(f"{result.tool_name} unavailable for query '{result.query}': {result.error}")
                continue

            for hit in result.qdrant_hits:
                key = f"qdrant:{hit.chunk_id}"
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    EvidenceItem(
                        citation_id=f"[E{qdrant_index}]",
                        source="qdrant",
                        chunk_id=hit.chunk_id,
                        doc_id=hit.doc_id,
                        title=hit.title,
                        article=hit.article_number,
                        clause=hit.clause_number,
                        text=hit.text,
                        context_text=hit.context_text,
                        score=hit.score,
                        source_url=hit.source_url,
                        effective_status=hit.tinh_trang_hieu_luc,
                    )
                )
                qdrant_index += 1

            for hit in result.web_hits:
                key = f"web:{hit.url}"
                if key in seen:
                    continue
                seen.add(key)
                text = hit.markdown or hit.description or hit.title
                items.append(
                    EvidenceItem(
                        citation_id=f"[W{web_index}]",
                        source="web",
                        title=hit.title,
                        text=text[:4000],
                        context_text=text[:4000],
                        score=hit.score,
                        source_url=hit.url,
                    )
                )
                web_index += 1

        sources_used = list(dict.fromkeys(item.source for item in items))
        return EvidencePacket(
            items=items[:10],
            coverage=self._coverage(items),
            sources_used=sources_used,
            notes=notes,
        )

    def _coverage(self, items: list[EvidenceItem]) -> str:
        qdrant_count = sum(1 for item in items if item.source == "qdrant")
        web_count = sum(1 for item in items if item.source == "web")
        if qdrant_count >= 3:
            return "sufficient"
        if qdrant_count >= 1 and (qdrant_count + web_count) >= 2:
            return "partial"
        if qdrant_count >= 1 or web_count >= 1:
            return "weak"
        return "none"

