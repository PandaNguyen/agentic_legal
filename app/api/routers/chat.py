from fastapi import APIRouter, Depends
from app.api.deps import get_legal_flow_runner, get_session_repo
from app.schemas.chat import ChatHistoryResponse, ChatRequest, ChatResponse
from app.services.orchestrator.legal_flow import LegalFlowRunner
from app.services.storage.session_repo import SessionRepository

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, flow_runner: LegalFlowRunner = Depends(get_legal_flow_runner)):
    return await flow_runner.run(req)


@router.get("/chat/sessions/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_session(
    session_id: str,
    session_repo: SessionRepository = Depends(get_session_repo),
):
    return ChatHistoryResponse(session_id=session_id, turns=await session_repo.get_history(session_id))


@router.delete("/chat/sessions/{session_id}", response_model=ChatHistoryResponse)
async def clear_chat_session(
    session_id: str,
    session_repo: SessionRepository = Depends(get_session_repo),
):
    await session_repo.clear_session(session_id)
    return ChatHistoryResponse(session_id=session_id, turns=[])
