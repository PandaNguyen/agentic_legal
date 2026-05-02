from typing import Any, Literal
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    audience: Literal["consumer", "business"] = "consumer"
    locale: str = "vi-VN"


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=3)
    search_web: bool = False
    stream: bool = False
    user_profile: UserProfile = UserProfile()


class Citation(BaseModel):
    doc_id: str
    doc_title: str
    article: str | None = None
    clause: str | None = None
    source_url: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    mode: Literal["faq", "scenario", "compare", "procedure", "fallback"]
    citations: list[Citation]
    disclaimer: str
    follow_up_questions: list[str] = []
    trace_id: str


class SessionTurn(BaseModel):
    user_message: str
    assistant_message: str
    trace_id: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    session_id: str
    turns: list[SessionTurn] = Field(default_factory=list)
