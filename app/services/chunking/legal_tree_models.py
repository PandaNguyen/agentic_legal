from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LegalTreeNode(BaseModel):
    node_id: str
    doc_id: str
    type: str
    number: str | None = None
    title: str | None = None
    text: str
    raw_text: str
    level: float
    parent_id: str | None = None
    path: list[str] = Field(default_factory=list)
    children: list["LegalTreeNode"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_edit: bool = False


class LegalDocumentTree(BaseModel):
    doc_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    parse_status: str = "ok"
    tree: LegalTreeNode
    stats: dict[str, int] = Field(default_factory=dict)


class ClosureEdge(BaseModel):
    doc_id: str
    ancestor: str
    descendant: str
    depth: int


class AdjacencyEdge(BaseModel):
    doc_id: str
    parent: str
    children: list[str] = Field(default_factory=list)
