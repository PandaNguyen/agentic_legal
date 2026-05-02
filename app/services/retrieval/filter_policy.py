from __future__ import annotations

from app.schemas.search import SearchFilters


CHECKFACETS_FIELDS = {
    "ngay_ban_hanh_ts",
    "effective_year",
    "so_ky_hieu_slug",
    "ngay_het_hieu_luc_ts",
    "expiry_year",
    "loai_van_ban",
    "so_ky_hieu",
    "nganh_slug",
    "issue_year",
    "nganh",
    "pham_vi_slug",
    "linh_vuc_slug",
    "pham_vi",
    "tinh_trang_hieu_luc",
    "linh_vuc",
    "loai_van_ban_slug",
    "tinh_trang_hieu_luc_slug",
    "co_quan_ban_hanh",
    "ngay_co_hieu_luc_ts",
    "co_quan_ban_hanh_slug",
}

ALLOWED_FILTER_FIELDS = tuple(SearchFilters.model_fields.keys())


def allowed_filter_fields() -> list[str]:
    return list(ALLOWED_FILTER_FIELDS)


def sanitize_search_filters(filters: SearchFilters | None) -> SearchFilters | None:
    if filters is None:
        return None

    data = filters.model_dump()
    sanitized: dict[str, object] = {}
    for field_name in ALLOWED_FILTER_FIELDS:
        value = data.get(field_name)
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            sanitized[field_name] = cleaned or None
        elif isinstance(value, str):
            text = value.strip()
            sanitized[field_name] = text or None
        else:
            sanitized[field_name] = value

    result = SearchFilters.model_validate(sanitized)
    return result if any(value is not None for value in result.model_dump().values()) else None
