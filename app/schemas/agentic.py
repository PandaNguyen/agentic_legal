from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.search import SearchFilters, SearchHit


ToolName = Literal["qdrant_search", "web_search"]
SourceName = Literal["qdrant", "web"]


class AgentDecision(BaseModel):
    agent_name: str
    reasoning_summary: str = ""
    action: str
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RouterDecision(BaseModel):
    intent: Literal["legal_qa", "scenario", "compare", "procedure", "out_of_scope"] = "legal_qa"
    sector: str | None = None
    answer_mode: Literal["faq", "scenario", "compare", "procedure", "fallback"] = "faq"
    risk_level: Literal["low", "medium", "high"] = "medium"
    facts_missing: list[str] = Field(default_factory=list)
    selected_tools: list[ToolName] = Field(default_factory=lambda: ["qdrant_search"])
    retrieval_queries: list[str] = Field(default_factory=list)
    filters: SearchFilters | None = None
    reasoning_summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class WebSearchHit(BaseModel):
    title: str
    url: str
    description: str | None = None
    markdown: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: ToolName
    query: str
    available: bool = True
    error: str | None = None
    qdrant_hits: list[SearchHit] = Field(default_factory=list)
    web_hits: list[WebSearchHit] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    citation_id: str
    source: SourceName
    chunk_id: str | None = None
    doc_id: str | None = None
    title: str
    article: str | None = None
    clause: str | None = None
    text: str
    context_text: str | None = None
    score: float | None = None
    source_url: str | None = None
    effective_status: str | None = None


class EvidencePacket(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)
    coverage: Literal["sufficient", "partial", "weak", "none"] = "none"
    sources_used: list[SourceName] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RetryPlan(BaseModel):
    rewritten_query: str = ""
    tool_name: Literal["qdrant_search", "web_search", "both"] = "qdrant_search"
    reason: str = ""


class VerifierDecision(BaseModel):
    action: Literal["accept", "revise_answer", "retry_retrieval", "ask_clarification"] = "accept"
    reasoning_summary: str = ""
    revised_answer: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    retry_plan: RetryPlan | None = None


class SessionTurn(BaseModel):
    user_message: str
    assistant_message: str
    trace_id: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)

