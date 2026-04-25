from app.schemas.chat import Citation, ChatResponse
from app.schemas.search import SearchHit
from app.services.llm.openai_client import OpenAIService

ANSWER_SYSTEM_PROMPT = """
You are a Vietnamese legal assistant for non-lawyers.
Rules:
- Use only the provided evidence.
- Be simple and practical.
- State uncertainty when evidence is incomplete.
- Do not claim to be a lawyer.
- Output Vietnamese.
""".strip()


class AnswerService:
    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service

    async def generate(
        self,
        message: str,
        mode: str,
        evidence: list[SearchHit],
        trace_id: str,
    ) -> ChatResponse:
        evidence_text = "\n\n".join(
            [
                f"[{idx+1}] {item.doc_title} | article={item.article} | clause={item.clause}\n{item.text}"
                for idx, item in enumerate(evidence)
            ]
        )
        user_prompt = f"Question:\n{message}\n\nEvidence:\n{evidence_text}"
        answer = await self.openai_service.generate_text(ANSWER_SYSTEM_PROMPT, user_prompt)

        citations = [
            Citation(
                doc_id=item.doc_id,
                doc_title=item.doc_title,
                article=item.article,
                clause=item.clause,
                source_url=item.source_url,
                score=item.score,
            )
            for item in evidence[:3]
        ]
        return ChatResponse(
            answer=answer,
            confidence=min(0.95, 0.45 + 0.1 * len(evidence)),
            mode=mode if mode in {"faq", "scenario", "compare", "procedure", "fallback"} else "fallback",
            citations=citations,
            disclaimer="Thông tin chỉ nhằm mục đích tham khảo pháp lý phổ thông, không thay thế tư vấn pháp lý chính thức.",
            follow_up_questions=[],
            trace_id=trace_id,
        )
