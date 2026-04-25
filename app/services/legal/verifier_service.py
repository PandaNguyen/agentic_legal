from app.schemas.chat import ChatResponse
from app.schemas.search import SearchHit
from app.services.llm.openai_client import OpenAIService

VERIFIER_SYSTEM_PROMPT = """
You verify a legal answer draft.
Check whether the answer overstates certainty relative to the evidence.
Return strict JSON with keys: revised_answer, confidence_adjustment, follow_up_questions.
Output Vietnamese for revised_answer and follow_up_questions.
""".strip()


class VerifierService:
    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service

    async def verify(self, draft: ChatResponse, evidence: list[SearchHit]) -> ChatResponse:
        evidence_text = "\n\n".join([f"- {item.doc_title}: {item.text}" for item in evidence[:5]])
        user_prompt = f"Draft answer:\n{draft.answer}\n\nEvidence:\n{evidence_text}"
        result = await self.openai_service.generate_json(VERIFIER_SYSTEM_PROMPT, user_prompt)
        draft.answer = result.get("revised_answer", draft.answer)
        draft.confidence = max(0.0, min(1.0, draft.confidence + float(result.get("confidence_adjustment", 0))))
        draft.follow_up_questions = result.get("follow_up_questions", [])
        return draft
