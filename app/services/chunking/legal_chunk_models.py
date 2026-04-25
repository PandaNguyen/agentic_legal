from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LegalChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_node_id: str
    source_node_type: str
    chunk_index: int
    text: str
    context_text: str
    tree_path: list[str] = Field(default_factory=list)
    content_path: list[str] = Field(default_factory=list)
    article_number: str | None = None
    article_title: str | None = None
    clause_number: str | None = None
    point_number: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
