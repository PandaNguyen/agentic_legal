from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.agents.answer_agent import AnswerAgent
from app.services.agents.router_agent import RouterAgent
from app.services.agents.runtime import AgentRuntime
from app.services.agents.verifier_agent import VerifierAgent
from app.services.legal.evidence_assembler import EvidenceAssemblerService
from app.services.llm.openai_client import OpenAIService
from app.services.orchestrator.legal_flow import LegalFlowRunner
from app.services.retrieval.qdrant_service import QdrantService
from app.services.storage.session_repo import SessionRepository
from app.services.tools.qdrant_search_tool import QdrantSearchTool
from app.services.tools.web_search_tool import WebSearchTool
from app.services.web.firecrawl_service import FirecrawlSearchService


_session_repo = SessionRepository(max_turns=10)


def get_openai_service(settings: Settings = Depends(get_settings)) -> OpenAIService:
    return OpenAIService(settings)


def get_agent_runtime(openai_service: OpenAIService = Depends(get_openai_service)) -> AgentRuntime:
    return AgentRuntime(openai_service)


def get_qdrant_service(settings: Settings = Depends(get_settings)) -> QdrantService:
    return QdrantService(settings)


def get_firecrawl_service(settings: Settings = Depends(get_settings)) -> FirecrawlSearchService:
    return FirecrawlSearchService(settings)


def get_qdrant_search_tool(qdrant_service: QdrantService = Depends(get_qdrant_service)) -> QdrantSearchTool:
    return QdrantSearchTool(qdrant_service)


def get_web_search_tool(firecrawl_service: FirecrawlSearchService = Depends(get_firecrawl_service)) -> WebSearchTool:
    return WebSearchTool(firecrawl_service)


def get_router_agent(runtime: AgentRuntime = Depends(get_agent_runtime)) -> RouterAgent:
    return RouterAgent(runtime)


def get_answer_agent(openai_service: OpenAIService = Depends(get_openai_service)) -> AnswerAgent:
    return AnswerAgent(openai_service)


def get_verifier_agent(runtime: AgentRuntime = Depends(get_agent_runtime)) -> VerifierAgent:
    return VerifierAgent(runtime)


def get_evidence_assembler() -> EvidenceAssemblerService:
    return EvidenceAssemblerService()


def get_session_repo() -> SessionRepository:
    return _session_repo


def get_legal_flow_runner(
    router_agent: RouterAgent = Depends(get_router_agent),
    qdrant_search_tool: QdrantSearchTool = Depends(get_qdrant_search_tool),
    web_search_tool: WebSearchTool = Depends(get_web_search_tool),
    evidence_assembler: EvidenceAssemblerService = Depends(get_evidence_assembler),
    answer_agent: AnswerAgent = Depends(get_answer_agent),
    verifier_agent: VerifierAgent = Depends(get_verifier_agent),
    session_repo: SessionRepository = Depends(get_session_repo),
) -> LegalFlowRunner:
    return LegalFlowRunner(
        router_agent=router_agent,
        qdrant_search_tool=qdrant_search_tool,
        web_search_tool=web_search_tool,
        evidence_assembler=evidence_assembler,
        answer_agent=answer_agent,
        verifier_agent=verifier_agent,
        session_repo=session_repo,
    )
