from fastapi import Depends
from app.core.config import Settings, get_settings
from app.services.llm.openai_client import OpenAIService
from app.services.retrieval.qdrant_service import QdrantService
from app.services.legal.router_service import RouterService
from app.services.legal.answer_service import AnswerService
from app.services.legal.verifier_service import VerifierService
from app.services.orchestrator.legal_flow import LegalFlowRunner


def get_openai_service(settings: Settings = Depends(get_settings)) -> OpenAIService:
    return OpenAIService(settings)


def get_qdrant_service(
    settings: Settings = Depends(get_settings),
) -> QdrantService:
    return QdrantService(settings)


def get_router_service(openai_service: OpenAIService = Depends(get_openai_service)) -> RouterService:
    return RouterService(openai_service)


def get_answer_service(openai_service: OpenAIService = Depends(get_openai_service)) -> AnswerService:
    return AnswerService(openai_service)


def get_verifier_service(openai_service: OpenAIService = Depends(get_openai_service)) -> VerifierService:
    return VerifierService(openai_service)


def get_legal_flow_runner(
    router_service: RouterService = Depends(get_router_service),
    qdrant_service: QdrantService = Depends(get_qdrant_service),
    answer_service: AnswerService = Depends(get_answer_service),
    verifier_service: VerifierService = Depends(get_verifier_service),
) -> LegalFlowRunner:
    return LegalFlowRunner(
        router_service=router_service,
        qdrant_service=qdrant_service,
        answer_service=answer_service,
        verifier_service=verifier_service,
    )
