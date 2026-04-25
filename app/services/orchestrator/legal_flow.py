from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.search import SearchFilters
from app.services.legal.router_service import RouterService
from app.services.legal.answer_service import AnswerService
from app.services.legal.verifier_service import VerifierService
from app.services.retrieval.qdrant_service import QdrantService

try:
    from crewai.flow.flow import Flow, start, listen
except Exception:  # pragma: no cover
    Flow = object
    def start():
        def decorator(fn):
            return fn
        return decorator
    def listen(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator


@dataclass
class LegalState:
    request: ChatRequest
    trace_id: str
    route: dict | None = None
    evidence: list | None = None
    draft: ChatResponse | None = None
    final: ChatResponse | None = None


class LegalFlow(Flow):
    def __init__(self, state: LegalState, router_service: RouterService, qdrant_service: QdrantService, answer_service: AnswerService, verifier_service: VerifierService):
        super().__init__()
        self.state = state
        self.router_service = router_service
        self.qdrant_service = qdrant_service
        self.answer_service = answer_service
        self.verifier_service = verifier_service

    @start()
    async def classify(self):
        self.state.route = await self.router_service.classify(
            self.state.request.message,
            self.state.request.search_web,
        )
        return self.state.route

    @listen(classify)
    async def retrieve(self, route: dict):
        sector = route.get("sector")
        filters = SearchFilters(nganh=[sector]) if sector else None
        self.state.evidence = await self.qdrant_service.search(
            query=self.state.request.message,
            top_k=5,
            filters=filters,
        )
        return self.state.evidence

    @listen(retrieve)
    async def reason(self, evidence):
        mode = (self.state.route or {}).get("answer_mode", "faq")
        self.state.draft = await self.answer_service.generate(
            message=self.state.request.message,
            mode=mode,
            evidence=evidence,
            trace_id=self.state.trace_id,
        )
        return self.state.draft

    @listen(reason)
    async def verify(self, draft):
        self.state.final = await self.verifier_service.verify(draft, self.state.evidence or [])
        return self.state.final


class LegalFlowRunner:
    def __init__(self, router_service: RouterService, qdrant_service: QdrantService, answer_service: AnswerService, verifier_service: VerifierService):
        self.router_service = router_service
        self.qdrant_service = qdrant_service
        self.answer_service = answer_service
        self.verifier_service = verifier_service

    async def run(self, req: ChatRequest) -> ChatResponse:
        trace_id = f"trc_{uuid.uuid4().hex[:12]}"
        state = LegalState(request=req, trace_id=trace_id)
        flow = LegalFlow(
            state=state,
            router_service=self.router_service,
            qdrant_service=self.qdrant_service,
            answer_service=self.answer_service,
            verifier_service=self.verifier_service,
        )
        await flow.kickoff_async()
        return state.final or state.draft
