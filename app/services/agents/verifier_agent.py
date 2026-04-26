from __future__ import annotations

from app.schemas.agentic import EvidencePacket, RetryPlan, RouterDecision, VerifierDecision
from app.schemas.chat import ChatResponse
from app.services.agents.runtime import AgentRuntime


VERIFIER_AGENT_PROMPT = """
You are VerifierAgent in a Vietnamese legal multi-agent RAG system.
You inspect an answer and decide the next action autonomously.

Allowed actions:
- accept: answer is sufficiently supported by evidence.
- revise_answer: answer has minor overstatements and can be fixed without more retrieval.
- retry_retrieval: evidence is weak/missing or the query should be rewritten.
- ask_clarification: user facts are too incomplete to answer safely.

Rules:
- Return strict JSON matching VerifierDecision.
- Choose retry_retrieval only when a better query or another allowed tool can materially improve the answer.
- If retry_retrieval is chosen, include retry_plan with rewritten_query, tool_name, and reason.
- Be conservative for legal claims.
- Do not expose hidden chain-of-thought; reasoning_summary must be brief.
""".strip()


class VerifierAgent:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def verify(
        self,
        *,
        message: str,
        route: RouterDecision,
        evidence: EvidencePacket,
        draft: ChatResponse,
        retry_count: int,
        max_retries: int = 1,
    ) -> VerifierDecision:
        fallback = VerifierDecision(
            action="accept" if evidence.coverage in {"sufficient", "partial"} else "revise_answer",
            revised_answer=None,
            confidence=draft.confidence,
            reasoning_summary="Fallback verifier decision based on evidence coverage.",
        )
        decision = await self.runtime.decide(
            system_prompt=VERIFIER_AGENT_PROMPT,
            user_context={
                "question": message,
                "route": route.model_dump(),
                "draft_answer": draft.answer,
                "draft_confidence": draft.confidence,
                "evidence": evidence.model_dump(),
                "retry_count": retry_count,
                "max_retries": max_retries,
            },
            output_schema=VerifierDecision,
            fallback=fallback,
        )
        if decision.action == "retry_retrieval" and retry_count >= max_retries:
            decision.action = "revise_answer"
            decision.retry_plan = None
            if not decision.revised_answer:
                decision.revised_answer = draft.answer
            decision.reasoning_summary = (
                f"{decision.reasoning_summary} Retry limit reached; revising without another retrieval."
            ).strip()
        if decision.action == "retry_retrieval" and decision.retry_plan is None:
            decision.retry_plan = RetryPlan(rewritten_query=message, tool_name="qdrant_search", reason="Retry requested")
        return decision

