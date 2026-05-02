from typing import Literal

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    doc_ids: list[str] | None = None
    so_ky_hieu: list[str] | None = None
    loai_van_ban: list[str] | None = None
    co_quan_ban_hanh: list[str] | None = None
    pham_vi: list[str] | None = None
    tinh_trang_hieu_luc: list[str] | None = None
    nganh: list[str] | None = None
    linh_vuc: list[str] | None = None
    issue_date_from: str | None = None
    issue_date_to: str | None = None
    effective_date_from: str | None = None
    effective_date_to: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    search_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    candidate_limit: int = 200
    filters: SearchFilters | None = None


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    article_number: str | None = None
    clause_number: str | None = None
    point_number: str | None = None
    source_node_type: str | None = None
    so_ky_hieu: str | None = None
    loai_van_ban: str | None = None
    co_quan_ban_hanh: str | None = None
    pham_vi: str | None = None
    tinh_trang_hieu_luc: str | None = None
    text: str
    context_text: str | None = None
    score: float
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)

    @property
    def doc_title(self) -> str:
        return self.title

    @property
    def article(self) -> str | None:
        return self.article_number

    @property
    def clause(self) -> str | None:
        return self.clause_number

    @property
    def source_url(self) -> str | None:
        value = self.metadata.get("source_url")
        return str(value) if value else None


class SearchResponse(BaseModel):
    results: list[SearchHit]
