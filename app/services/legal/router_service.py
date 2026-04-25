from app.services.llm.openai_client import OpenAIService

ROUTER_SYSTEM_PROMPT = """
You are a legal request router for a Vietnam legal assistant.
Return strict JSON with keys:
intent, sector, needs_web, risk_level, answer_mode, facts_missing.
Possible intent: legal_qa, scenario, compare, procedure, out_of_scope.
Possible answer_mode: faq, scenario, compare, procedure, fallback.
Keep sector short, snake_case.
""".strip()


class RouterService:
    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service

    async def classify(self, message: str, search_web: bool) -> dict:
        user_prompt = f"message={message}\nsearch_web_enabled={search_web}"
        data = await self.openai_service.generate_json(ROUTER_SYSTEM_PROMPT, user_prompt)
        data.setdefault("answer_mode", "faq")
        data.setdefault("needs_web", False)
        data.setdefault("facts_missing", [])
        return data
