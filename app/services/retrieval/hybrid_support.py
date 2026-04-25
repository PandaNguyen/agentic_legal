from __future__ import annotations

import re
import unicodedata
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Protocol

from qdrant_client import models

from app.services.chunking.legal_chunk_models import LegalChunk

if TYPE_CHECKING:
    from app.schemas.search import SearchFilters


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_PIPELINE_VERSION = "hybrid_v1"
MAX_RELATED_DOC_IDS = 16
TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[./_:-][0-9A-Za-zÀ-ỹĐđ]+)*", flags=re.UNICODE)
DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y")

KEYWORD_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "chunk_id": models.PayloadSchemaType.KEYWORD,
    "so_ky_hieu": models.PayloadSchemaType.KEYWORD,
    "so_ky_hieu_slug": models.PayloadSchemaType.KEYWORD,
    "loai_van_ban": models.PayloadSchemaType.KEYWORD,
    "loai_van_ban_slug": models.PayloadSchemaType.KEYWORD,
    "co_quan_ban_hanh": models.PayloadSchemaType.KEYWORD,
    "co_quan_ban_hanh_slug": models.PayloadSchemaType.KEYWORD,
    "pham_vi": models.PayloadSchemaType.KEYWORD,
    "pham_vi_slug": models.PayloadSchemaType.KEYWORD,
    "tinh_trang_hieu_luc": models.PayloadSchemaType.KEYWORD,
    "tinh_trang_hieu_luc_slug": models.PayloadSchemaType.KEYWORD,
    "nganh": models.PayloadSchemaType.KEYWORD,
    "nganh_slug": models.PayloadSchemaType.KEYWORD,
    "linh_vuc": models.PayloadSchemaType.KEYWORD,
    "linh_vuc_slug": models.PayloadSchemaType.KEYWORD,
    "source_node_type": models.PayloadSchemaType.KEYWORD,
}
INTEGER_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "chunk_index": models.PayloadSchemaType.INTEGER,
    "token_count": models.PayloadSchemaType.INTEGER,
    "relationship_count": models.PayloadSchemaType.INTEGER,
    "ngay_ban_hanh_ts": models.PayloadSchemaType.INTEGER,
    "ngay_co_hieu_luc_ts": models.PayloadSchemaType.INTEGER,
    "ngay_het_hieu_luc_ts": models.PayloadSchemaType.INTEGER,
    "issue_year": models.PayloadSchemaType.INTEGER,
    "effective_year": models.PayloadSchemaType.INTEGER,
    "expiry_year": models.PayloadSchemaType.INTEGER,
}


class TermIdResolver(Protocol):
    def resolve_term_ids(self, terms: Iterable[str], create_missing: bool = False) -> dict[str, int]:
        ...


class SentenceTransformerDenseEncoder:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def embedding_dimension(self) -> int:
        model = self._get_model()
        dimension = model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Unable to determine embedding dimension for model {self.model_name}")
        return int(dimension)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode_document(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )
        return embeddings.tolist()

    def encode_query(self, text: str) -> list[float]:
        model = self._get_model()
        embedding = model.encode_query(
            text,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("sentence-transformers is required for hybrid retrieval/ingest") from exc

        model_kwargs: dict[str, Any] = {}
        if self.use_fp16 and self.device and self.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = "float16"

        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=model_kwargs or None,
            )
        except TypeError:
            self._model = SentenceTransformer(self.model_name, device=self.device)

        if self.use_fp16 and self.device and self.device.startswith("cuda") and hasattr(self._model, "half"):
            self._model.half()

        return self._model


def normalize_sparse_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip().lower()


def tokenize_sparse_text(text: str) -> list[str]:
    normalized = normalize_sparse_text(text)
    return [match.group(0) for match in TOKEN_PATTERN.finditer(normalized)]


def resolve_term_frequencies(text: str) -> Counter[str]:
    return Counter(tokenize_sparse_text(text))


def build_sparse_vector(
    text: str,
    resolver: TermIdResolver,
    *,
    create_missing: bool,
) -> models.SparseVector | None:
    frequencies = resolve_term_frequencies(text)
    if not frequencies:
        return None

    term_ids = resolver.resolve_term_ids(frequencies.keys(), create_missing=create_missing)
    items = sorted(
        (term_ids[token], float(frequencies[token]))
        for token in frequencies
        if token in term_ids
    )
    if not items:
        return None
    indices = [term_id for term_id, _ in items]
    values = [value for _, value in items]
    return models.SparseVector(indices=indices, values=values)


def deterministic_point_id(chunk_id: str, pipeline_version: str = DEFAULT_PIPELINE_VERSION) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pipeline_version}:{chunk_id}"))


def normalize_filter_keyword(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKD", str(value).strip().lower())
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    slug = re.sub(r"[^0-9a-z]+", "_", without_marks).strip("_")
    return slug or None


def parse_legal_date(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "..."}:
        return None
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def datetime_to_timestamp(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp())


def build_chunk_payload(chunk: LegalChunk, pipeline_version: str = DEFAULT_PIPELINE_VERSION) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    relationships = chunk.relationships or []
    issue_dt = parse_legal_date(metadata.get("ngay_ban_hanh"))
    effective_dt = parse_legal_date(metadata.get("ngay_co_hieu_luc"))
    expiry_dt = parse_legal_date(metadata.get("ngay_het_hieu_luc"))
    relationship_labels = sorted({str(item.get("relationship")).strip() for item in relationships if item.get("relationship")})
    related_doc_ids = list(
        dict.fromkeys(
            str(item.get("other_doc_id")).strip()
            for item in relationships
            if item.get("other_doc_id") is not None and str(item.get("other_doc_id")).strip()
        )
    )[:MAX_RELATED_DOC_IDS]

    payload = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "source_node_type": chunk.source_node_type,
        "source_node_id": chunk.source_node_id,
        "article_number": chunk.article_number,
        "article_title": chunk.article_title,
        "clause_number": chunk.clause_number,
        "point_number": chunk.point_number,
        "token_count": chunk.token_count,
        "pipeline_version": pipeline_version,
        "text": chunk.text,
        "context_text": chunk.context_text,
        "title": clean_optional(metadata.get("title")),
        "doc_title": clean_optional(metadata.get("title")),
        "so_ky_hieu": clean_optional(metadata.get("so_ky_hieu")),
        "loai_van_ban": clean_optional(metadata.get("loai_van_ban")),
        "ngay_ban_hanh": clean_optional(metadata.get("ngay_ban_hanh")),
        "ngay_co_hieu_luc": clean_optional(metadata.get("ngay_co_hieu_luc")),
        "ngay_het_hieu_luc": clean_optional(metadata.get("ngay_het_hieu_luc")),
        "nganh": clean_optional(metadata.get("nganh")),
        "linh_vuc": clean_optional(metadata.get("linh_vuc")),
        "co_quan_ban_hanh": clean_optional(metadata.get("co_quan_ban_hanh")),
        "pham_vi": clean_optional(metadata.get("pham_vi")),
        "tinh_trang_hieu_luc": clean_optional(metadata.get("tinh_trang_hieu_luc")),
        "source_url": clean_optional(metadata.get("source_url")),
        "ngay_ban_hanh_ts": datetime_to_timestamp(issue_dt),
        "ngay_co_hieu_luc_ts": datetime_to_timestamp(effective_dt),
        "ngay_het_hieu_luc_ts": datetime_to_timestamp(expiry_dt),
        "issue_year": issue_dt.year if issue_dt else None,
        "effective_year": effective_dt.year if effective_dt else None,
        "expiry_year": expiry_dt.year if expiry_dt else None,
        "relationship_labels": relationship_labels,
        "relationship_count": len(relationships),
        "related_doc_ids": related_doc_ids,
        "so_ky_hieu_slug": normalize_filter_keyword(clean_optional(metadata.get("so_ky_hieu"))),
        "loai_van_ban_slug": normalize_filter_keyword(clean_optional(metadata.get("loai_van_ban"))),
        "co_quan_ban_hanh_slug": normalize_filter_keyword(clean_optional(metadata.get("co_quan_ban_hanh"))),
        "pham_vi_slug": normalize_filter_keyword(clean_optional(metadata.get("pham_vi"))),
        "tinh_trang_hieu_luc_slug": normalize_filter_keyword(clean_optional(metadata.get("tinh_trang_hieu_luc"))),
        "nganh_slug": normalize_filter_keyword(clean_optional(metadata.get("nganh"))),
        "linh_vuc_slug": normalize_filter_keyword(clean_optional(metadata.get("linh_vuc"))),
    }
    return compact_payload(payload)


def build_qdrant_filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None:
        return None

    must: list[models.FieldCondition] = []
    add_match_conditions(must, "doc_id", filters.doc_ids)
    add_slug_match_conditions(must, "so_ky_hieu_slug", filters.so_ky_hieu)
    add_slug_match_conditions(must, "loai_van_ban_slug", filters.loai_van_ban)
    add_slug_match_conditions(must, "co_quan_ban_hanh_slug", filters.co_quan_ban_hanh)
    add_slug_match_conditions(must, "pham_vi_slug", filters.pham_vi)
    add_slug_match_conditions(must, "tinh_trang_hieu_luc_slug", filters.tinh_trang_hieu_luc)
    add_slug_match_conditions(must, "nganh_slug", filters.nganh)
    add_slug_match_conditions(must, "linh_vuc_slug", filters.linh_vuc)
    add_date_range_conditions(must, "ngay_ban_hanh_ts", filters.issue_date_from, filters.issue_date_to)
    add_date_range_conditions(must, "ngay_co_hieu_luc_ts", filters.effective_date_from, filters.effective_date_to)
    return models.Filter(must=must) if must else None


def add_match_conditions(
    must: list[models.FieldCondition],
    key: str,
    values: list[str] | None,
) -> None:
    cleaned = [str(value).strip() for value in values or [] if str(value).strip()]
    if not cleaned:
        return
    if len(cleaned) == 1:
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=cleaned[0])))
        return
    must.append(models.FieldCondition(key=key, match=models.MatchAny(any=cleaned)))


def add_slug_match_conditions(
    must: list[models.FieldCondition],
    key: str,
    values: list[str] | None,
) -> None:
    normalized = [normalize_filter_keyword(value) for value in values or []]
    add_match_conditions(must, key, [value for value in normalized if value])


def add_date_range_conditions(
    must: list[models.FieldCondition],
    key: str,
    start: str | None,
    end: str | None,
) -> None:
    start_ts = datetime_to_timestamp(parse_legal_date(start))
    end_ts = datetime_to_timestamp(parse_legal_date(end))
    if start_ts is None and end_ts is None:
        return
    must.append(models.FieldCondition(key=key, range=models.Range(gte=start_ts, lte=end_ts)))


def payload_index_specs() -> dict[str, models.PayloadSchemaType]:
    return {**KEYWORD_PAYLOAD_INDEXES, **INTEGER_PAYLOAD_INDEXES}


def clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        compacted[key] = value
    return compacted
