from __future__ import annotations

from app.schemas.agentic import RouterDecision
from app.schemas.chat import ChatRequest
from app.services.retrieval.filter_policy import allowed_filter_fields, sanitize_search_filters
from app.services.agents.runtime import AgentRuntime


ROUTER_AGENT_PROMPT = """
You are RouterAgent in a multi-agent Agentic RAG system for Vietnamese law.
You must choose retrieval tools autonomously from the allowed tool set.

Allowed selected_tools values:
- qdrant_search: use the local ingested legal corpus.
- web_search: use trusted live legal web search when currency/effectiveness must be checked.

Rules:
- Return strict JSON matching the RouterDecision schema.
- Do not use if/else style labels. Explain the action in reasoning_summary.
- Use qdrant_search for normal legal questions answerable from the local corpus.
- Use web_search when the user asks for latest/current updates, effectivity checks, or external source validation.
- Use both tools when local evidence should be combined with current validation.
- If facts are missing, list them but still choose retrieval unless retrieval would be misleading.
- Keep retrieval_queries short and directly searchable.
- Only produce filters from allowed_filter_fields in the user context. Ignore uncertain filters.
- Output field names exactly: intent, sector, answer_mode, risk_level, facts_missing, selected_tools, retrieval_queries, filters, reasoning_summary, confidence.
""".strip()


class RouterAgent:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def decide(self, request: ChatRequest, session_history: list[dict]) -> RouterDecision:
        fallback = RouterDecision(
            selected_tools=["qdrant_search"],
            retrieval_queries=[request.message],
            reasoning_summary="Fallback to local legal corpus search because routing output was unavailable.",
            confidence=0.4,
        )
        decision = await self.runtime.decide(
            system_prompt=ROUTER_AGENT_PROMPT,
            user_context={
                "message": request.message,
                "search_web_requested": request.search_web,
                "user_profile": request.user_profile.model_dump(),
                "session_history": session_history[-3:],
                "available_tools": ["qdrant_search", "web_search"],
                "allowed_filter_fields": allowed_filter_fields(),
            },
            output_schema=RouterDecision,
            fallback=fallback,
        )
        if not decision.retrieval_queries:
            decision.retrieval_queries = [request.message]
        if not decision.selected_tools:
            decision.selected_tools = ["qdrant_search"]
        decision.filters = sanitize_search_filters(decision.filters)
        return decision
