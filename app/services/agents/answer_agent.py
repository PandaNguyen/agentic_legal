from __future__ import annotations

from app.schemas.agentic import EvidencePacket, RouterDecision
from app.schemas.chat import ChatResponse, Citation
from app.services.llm.openai_client import OpenAIService


ANSWER_AGENT_PROMPT = """
You are AnswerAgent in a Vietnamese legal multi-agent RAG system.
Answer in Vietnamese for non-lawyers.

Rules:
- Use only the provided evidence items.
- Cite evidence ids such as [E1] or [W1] in the answer when making legal claims.
- If evidence coverage is weak or none, say that the available evidence is insufficient.
- Do not claim to be a lawyer.
- Be practical and concise.
""".strip()


DISCLAIMER = (
    "Thong tin chi nham muc dich tham khao phap ly pho thong, "
    "khong thay the tu van phap ly chinh thuc."
)


class AnswerAgent:
    def __init__(self, openai_service: OpenAIService) -> None:
        self.openai_service = openai_service

    async def generate(
        self,
        *,
        message: str,
        route: RouterDecision,
        evidence: EvidencePacket,
        trace_id: str,
        session_history: list[dict] | None = None,
    ) -> ChatResponse:
        if not evidence.items:
            return ChatResponse(
                answer=(
                    "Toi chua tim thay can cu phap ly du de tra loi chac chan. "
                    "Ban co the bo sung linh vuc, thoi diem ap dung, dia phuong hoac van ban lien quan."
                ),
                confidence=0.2,
                mode=route.answer_mode,
                citations=[],
                disclaimer=DISCLAIMER,
                follow_up_questions=route.facts_missing,
                trace_id=trace_id,
            )

        evidence_text = "\n\n".join(
            [
                (
                    f"{item.citation_id} source={item.source} title={item.title} "
                    f"article={item.article} clause={item.clause} url={item.source_url}\n"
                    f"{item.context_text or item.text}"
                )
                for item in evidence.items
            ]
        )
        history_text = self._format_history(session_history or [])
        user_prompt = (
            f"Question:\n{message}\n\n"
            f"Conversation history for resolving references only:\n{history_text}\n\n"
            f"Answer mode: {route.answer_mode}\n"
            f"Evidence coverage: {evidence.coverage}\n"
            f"Facts missing: {route.facts_missing}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        answer = await self.openai_service.generate_text(ANSWER_AGENT_PROMPT, user_prompt)
        return ChatResponse(
            answer=answer,
            confidence=self._initial_confidence(evidence.coverage),
            mode=route.answer_mode,
            citations=self._build_citations(evidence),
            disclaimer=DISCLAIMER,
            follow_up_questions=route.facts_missing,
            trace_id=trace_id,
        )

    def _format_history(self, session_history: list[dict]) -> str:
        if not session_history:
            return "No prior turns."
        lines: list[str] = []
        for idx, turn in enumerate(session_history[-3:], start=1):
            user_message = str(turn.get("user_message") or "").strip()
            assistant_message = str(turn.get("assistant_message") or "").strip()
            lines.append(f"[Turn {idx}] User: {user_message}\nAssistant: {assistant_message}")
        return "\n\n".join(lines)

    def _initial_confidence(self, coverage: str) -> float:
        if coverage == "sufficient":
            return 0.75
        if coverage == "partial":
            return 0.6
        if coverage == "weak":
            return 0.4
        return 0.2

    def _build_citations(self, evidence: EvidencePacket) -> list[Citation]:
        citations: list[Citation] = []
        for item in evidence.items[:5]:
            citations.append(
                Citation(
                    doc_id=item.doc_id or item.source_url or item.citation_id,
                    doc_title=item.title,
                    article=item.article,
                    clause=item.clause,
                    source_url=item.source_url,
                    score=item.score,
                )
            )
        return citations
