from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.schemas.agentic import EvidencePacket, RetryPlan, RouterDecision, ToolName, ToolResult, VerifierDecision
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agents.answer_agent import DISCLAIMER, AnswerAgent
from app.services.agents.router_agent import RouterAgent
from app.services.agents.verifier_agent import VerifierAgent
from app.services.legal.evidence_assembler import EvidenceAssemblerService
from app.services.storage.session_repo import SessionRepository
from app.services.tools.qdrant_search_tool import QdrantSearchTool
from app.services.tools.web_search_tool import DEFAULT_LEGAL_DOMAINS, WebSearchTool


MAX_VERIFIER_RETRIES = 1


@dataclass
class LegalState:
    request: ChatRequest
    trace_id: str
    session_history: list[dict] = field(default_factory=list)
    route: RouterDecision | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    evidence: EvidencePacket | None = None
    draft: ChatResponse | None = None
    verifier_decision: VerifierDecision | None = None
    final: ChatResponse | None = None
    retry_count: int = 0


class LegalFlowRunner:
    def __init__(
        self,
        *,
        router_agent: RouterAgent,
        qdrant_search_tool: QdrantSearchTool,
        web_search_tool: WebSearchTool,
        evidence_assembler: EvidenceAssemblerService,
        answer_agent: AnswerAgent,
        verifier_agent: VerifierAgent,
        session_repo: SessionRepository,
    ) -> None:
        self.router_agent = router_agent
        self.qdrant_search_tool = qdrant_search_tool
        self.web_search_tool = web_search_tool
        self.evidence_assembler = evidence_assembler
        self.answer_agent = answer_agent
        self.verifier_agent = verifier_agent
        self.session_repo = session_repo

    async def run(self, req: ChatRequest) -> ChatResponse:
        state = LegalState(request=req, trace_id=f"trc_{uuid.uuid4().hex[:12]}")
        state.session_history = await self.session_repo.get_history(req.session_id)
        state.route = await self.router_agent.decide(req, state.session_history)
        state.tool_results = await self._run_router_tools(state.route, req.message)
        state.evidence = self.evidence_assembler.assemble(state.tool_results)
        state.draft = await self.answer_agent.generate(
            message=req.message,
            route=state.route,
            evidence=state.evidence,
            trace_id=state.trace_id,
            session_history=state.session_history,
        )
        state.verifier_decision = await self.verifier_agent.verify(
            message=req.message,
            route=state.route,
            evidence=state.evidence,
            draft=state.draft,
            retry_count=state.retry_count,
            max_retries=MAX_VERIFIER_RETRIES,
        )

        if state.verifier_decision.action == "retry_retrieval" and state.verifier_decision.retry_plan:
            await self._retry_once(state, state.verifier_decision.retry_plan)

        state.final = self._finalize(state.draft, state.verifier_decision, state.evidence)
        await self.session_repo.save_turn(
            req.session_id,
            req.message,
            state.final.answer,
            trace_id=state.trace_id,
            citations=[citation.model_dump() for citation in state.final.citations],
        )
        return state.final

    async def _retry_once(self, state: LegalState, retry_plan: RetryPlan) -> None:
        state.retry_count += 1
        retry_results = await self._run_retry_tools(retry_plan)
        state.tool_results.extend(retry_results)
        state.evidence = self.evidence_assembler.assemble(state.tool_results)
        assert state.route is not None
        state.draft = await self.answer_agent.generate(
            message=state.request.message,
            route=state.route,
            evidence=state.evidence,
            trace_id=state.trace_id,
            session_history=state.session_history,
        )
        state.verifier_decision = await self.verifier_agent.verify(
            message=state.request.message,
            route=state.route,
            evidence=state.evidence,
            draft=state.draft,
            retry_count=state.retry_count,
            max_retries=MAX_VERIFIER_RETRIES,
        )

    async def _run_router_tools(self, route: RouterDecision, original_query: str) -> list[ToolResult]:
        queries = route.retrieval_queries or [original_query]
        selected_tools = route.selected_tools or ["qdrant_search"]
        results: list[ToolResult] = []
        for query in queries[:3]:
            for tool_name in selected_tools:
                results.append(await self._run_tool(tool_name, query=query, route=route))
        return results

    async def _run_retry_tools(self, retry_plan: RetryPlan) -> list[ToolResult]:
        if retry_plan.tool_name == "both":
            tool_names: list[ToolName] = ["qdrant_search", "web_search"]
        else:
            tool_names = [retry_plan.tool_name]
        return [
            await self._run_tool(tool_name, query=retry_plan.rewritten_query or retry_plan.reason)
            for tool_name in tool_names
        ]

    async def _run_tool(
        self,
        tool_name: ToolName,
        *,
        query: str,
        route: RouterDecision | None = None,
    ) -> ToolResult:
        if tool_name == "web_search":
            return await self.web_search_tool.run(
                query=query,
                allowed_domains=DEFAULT_LEGAL_DOMAINS,
                max_results=5,
            )
        return await self.qdrant_search_tool.run(
            query=query,
            search_mode="hybrid",
            top_k=8,
            candidate_limit=200,
            filters=route.filters if route else None,
        )

    def _finalize(
        self,
        draft: ChatResponse | None,
        verifier_decision: VerifierDecision | None,
        evidence: EvidencePacket | None,
    ) -> ChatResponse:
        if draft is None:
            return ChatResponse(
                answer="Toi chua tao duoc cau tra loi tu cac nguon hien co.",
                confidence=0.0,
                mode="fallback",
                citations=[],
                disclaimer=DISCLAIMER,
                follow_up_questions=[],
                trace_id="",
            )
        if verifier_decision is None:
            return draft

        if verifier_decision.action == "ask_clarification":
            draft.answer = verifier_decision.revised_answer or (
                "Can bo sung them thong tin truoc khi co the dua ra cau tra loi phap ly dang tin cay."
            )
        elif verifier_decision.action == "revise_answer" and verifier_decision.revised_answer:
            draft.answer = verifier_decision.revised_answer

        draft.confidence = verifier_decision.confidence
        if verifier_decision.follow_up_questions:
            draft.follow_up_questions = verifier_decision.follow_up_questions
        if evidence and evidence.coverage in {"weak", "none"} and not draft.follow_up_questions:
            draft.follow_up_questions = ["Ban co the bo sung them thong tin ve thoi diem, dia phuong hoac van ban lien quan khong?"]
        return draft
