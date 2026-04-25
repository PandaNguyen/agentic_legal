from fastapi import APIRouter, Depends
from app.api.deps import get_legal_flow_runner
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.orchestrator.legal_flow import LegalFlowRunner

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, flow_runner: LegalFlowRunner = Depends(get_legal_flow_runner)):
    return await flow_runner.run(req)
