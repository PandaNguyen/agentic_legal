from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.chunking.legal_chunk_models import LegalChunk


# STRUCTURAL_TYPES = {"part", "appendix", "chapter", "section", "article", "clause", "point"}
# FORCED_CHUNK_TYPES = {"clause", "point", "table"}
STRUCTURAL_TYPES = {"part", "appendix", "chapter", "section", "article", "clause", "point"}
FORCED_CHUNK_TYPES = {"clause", "point", "table"}

# Node có vai trò gom nội dung lớn, không nên tự compact nguyên subtree.
CONTAINER_TYPES = {"part", "appendix", "chapter", "section"}

# # Node nên được ưu tiên làm chunk compact nếu đủ nhỏ.
# COMPACT_SUBTREE_TYPES = {"article", "clause", "point"}

@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 450
    chunk_overlap: int = 64
    table_chunk_size: int = 450
    table_header_rows: int = 3
    token_encoding_name: str = "cl100k_base"
    min_chunk_tokens: int = 24
    include_relationships: bool = True

class LegalTreeChunkExtractor:
    """Create retrieval chunks from legal document trees.

    This ports the core idea from ChunkExtractor: traverse the legal tree,
    create compact subtree chunks when possible, split long text, and split
    long tables by rows while preserving table headers.
    """

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self.encoding = _get_encoding(self.config.token_encoding_name)
        self.text_splitter = _get_text_splitter(
            self.config.token_encoding_name,
            self.config.chunk_size,
            _effective_overlap(self.config.chunk_size, self.config.chunk_overlap),
        )

    def extract_document_chunks(self, document_tree: dict[str, Any]) -> list[LegalChunk]:
        doc_id = str(document_tree["doc_id"])
        root = document_tree["tree"]
        metadata = document_tree.get("metadata") or {}
        relationships = document_tree.get("relationships") or []
        doc_context = _document_context(metadata)
        chunks: list[LegalChunk] = []

        for child in root.get("children", []):
            self._extract_node(
                node=child,
                doc_id=doc_id,
                doc_context=doc_context,
                metadata=metadata,
                relationships=relationships,
                ancestor_path=[],
                current_article=None,
                current_article_title=None,
                chunks=chunks,
            )

        chunks = self._merge_adjacent_content_chunks(chunks)
        return [
            chunk.model_copy(update={"chunk_index": index, "chunk_id": f"{doc_id}:chunk:{index}"})
            for index, chunk in enumerate(chunks)
        ]

    def _extract_node(
        self,
        node: dict[str, Any],
        doc_id: str,
        doc_context: str,
        metadata: dict[str, Any],
        relationships: list[dict[str, Any]],
        ancestor_path: list[str],
        current_article: str | None,
        current_article_title: str | None,
        chunks: list[LegalChunk],
    ) -> None:
        node_type = node.get("type") or "content"
        node_text = _clean_text(node.get("text") or "")
        node_label = _node_label(node)
        content_path = [*ancestor_path, node_label] if node_label else ancestor_path

        article_number = current_article
        article_title = current_article_title
        if node_type == "article":
            article_number = _clean_optional(node.get("number"))
            article_title = _clean_optional(node.get("title"))

        subtree_text = _merge_subtree_text(node)
        subtree_token_count = count_tokens(subtree_text)

        if not node_text and not subtree_text:
            return

        if node_type == "table":
            for text in self._split_table(subtree_text):
                self._append_chunk(
                    chunks=chunks,
                    doc_id=doc_id,
                    node=node,
                    text=text,
                    doc_context=doc_context,
                    content_path=content_path,
                    metadata=metadata,
                    relationships=relationships,
                    article_number=article_number,
                    article_title=article_title,
                )
            return

        if node_type in FORCED_CHUNK_TYPES:
            for text in self._split_text_if_needed(subtree_text):
                self._append_chunk(
                    chunks=chunks,
                    doc_id=doc_id,
                    node=node,
                    text=text,
                    doc_context=doc_context,
                    content_path=content_path,
                    metadata=metadata,
                    relationships=relationships,
                    article_number=article_number,
                    article_title=article_title,
                )
            return

        if subtree_token_count <= self.config.chunk_size:
            self._append_chunk(
                chunks=chunks,
                doc_id=doc_id,
                node=node,
                text=subtree_text,
                doc_context=doc_context,
                content_path=content_path,
                metadata=metadata,
                relationships=relationships,
                article_number=article_number,
                article_title=article_title,
            )
            return

        children = node.get("children") or []
        if not children:
            for text in self._split_text_if_needed(subtree_text):
                self._append_chunk(
                    chunks=chunks,
                    doc_id=doc_id,
                    node=node,
                    text=text,
                    doc_context=doc_context,
                    content_path=content_path,
                    metadata=metadata,
                    relationships=relationships,
                    article_number=article_number,
                    article_title=article_title,
                )
            return

        title_child_id = None

        if node_text and node_type in STRUCTURAL_TYPES:
            heading_text, title_child = _structural_heading_with_title_child(node)

            if title_child is not None:
                title_child_id = id(title_child)

            if _is_rich_heading(heading_text):
                self._append_chunk(
                    chunks=chunks,
                    doc_id=doc_id,
                    node=node,
                    text=heading_text,
                    doc_context=doc_context,
                    content_path=content_path,
                    metadata=metadata,
                    relationships=relationships,
                    article_number=article_number,
                    article_title=article_title,
                )

        for child in children:
            if title_child_id is not None and id(child) == title_child_id:
                continue

            child_ancestor_path = content_path

            if title_child_id is not None:
                title_text = _clean_text(title_child.get("text") or "")
                if title_text and title_text not in child_ancestor_path:
                    child_ancestor_path = [*content_path, title_text]

            self._extract_node(
                node=child,
                doc_id=doc_id,
                doc_context=doc_context,
                metadata=metadata,
                relationships=relationships,
                ancestor_path=child_ancestor_path,
                current_article=article_number,
                current_article_title=article_title,
                chunks=chunks,
            )


    def _append_chunk(
        self,
        chunks: list[LegalChunk],
        doc_id: str,
        node: dict[str, Any],
        text: str,
        doc_context: str,
        content_path: list[str],
        metadata: dict[str, Any],
        relationships: list[dict[str, Any]],
        article_number: str | None,
        article_title: str | None,
    ) -> None:
        cleaned_text = _clean_text(text)
        if not cleaned_text:
            return
        node_type = node.get("type") or "content"
        if self._is_low_value_chunk(cleaned_text, node_type, article_number):
            return

        # context_lines = [line for line in [doc_context, *content_path[:-1]] if line]
        # context_text = "\n".join([*context_lines, cleaned_text]).strip()

        context_lines = [doc_context, *content_path]

        if content_path and cleaned_text.startswith(content_path[-1]):
            context_lines = [doc_context, *content_path[:-1]]

        context_text = "\n".join([*context_lines, cleaned_text]).strip()
        chunk_metadata = _chunk_metadata(metadata)

        chunks.append(
            LegalChunk(
                chunk_id="",
                doc_id=doc_id,
                source_node_id=str(node.get("node_id")),
                source_node_type=node_type,
                chunk_index=-1,
                text=cleaned_text,
                context_text=context_text,
                tree_path=list(node.get("path") or []),
                content_path=content_path,
                article_number=article_number,
                article_title=article_title,
                clause_number=_clean_optional(node.get("number")) if node_type == "clause" else None,
                point_number=_clean_optional(node.get("number")) if node_type == "point" else None,
                token_count=self._count_tokens(context_text),
                metadata=chunk_metadata,
                relationships=relationships if self.config.include_relationships else [],
            )
        )

    def _split_text_if_needed(self, text: str) -> list[str]:
        if self._count_tokens(text) <= self.config.chunk_size:
            return [text]
        return [chunk for chunk in self.text_splitter.split_text(text) if _clean_text(chunk)]

    def _split_table(self, table: str) -> list[str]:
        if self._count_tokens(table) <= self.config.table_chunk_size:
            return [table]

        lines = [line for line in table.splitlines() if line.strip()]
        if not lines:
            return []

        header_rows = max(1, self.config.table_header_rows)
        header = lines[:header_rows]
        if self._count_tokens("\n".join(header)) > self.config.table_chunk_size / 2 and header_rows > 1:
            header_rows = 2 if len(lines) > 1 and _is_markdown_table_separator(lines[1]) else 1
            header = lines[:header_rows]

        rows = lines[header_rows:]
        chunks: list[str] = []
        current_rows: list[str] = []

        for row in rows:
            candidate = "\n".join([*header, *current_rows, row])
            if current_rows and self._count_tokens(candidate) > self.config.table_chunk_size:
                chunks.append("\n".join([*header, *current_rows]))
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows:
            chunks.append("\n".join([*header, *current_rows]))
        elif header:
            chunks.append("\n".join(header))
        return chunks or [table]

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    def _is_low_value_chunk(self, text: str, node_type: str, article_number: str | None) -> bool:
        if _is_decorative_text(text):
            return True
        if node_type != "content" or article_number:
            return False
        if self._count_tokens(text) >= self.config.min_chunk_tokens:
            return False
        return not _looks_like_legal_heading(text)

    def _merge_adjacent_content_chunks(self, chunks: list[LegalChunk]) -> list[LegalChunk]:
        merged: list[LegalChunk] = []
        pending: LegalChunk | None = None

        for chunk in chunks:
            if not self._can_merge_content_chunk(chunk):
                if pending is not None:
                    merged.append(pending)
                    pending = None
                merged.append(chunk)
                continue

            if pending is None:
                pending = chunk
                continue

            candidate_text = "\n".join([pending.text, chunk.text]).strip()
            candidate_context = _replace_context_text(pending.context_text, pending.text, candidate_text)
            same_parent = pending.tree_path[:-1] == chunk.tree_path[:-1]
            if same_parent and self._count_tokens(candidate_context) <= self.config.chunk_size:
                pending = pending.model_copy(
                    update={
                        "text": candidate_text,
                        "context_text": candidate_context,
                        "token_count": self._count_tokens(candidate_context),
                    }
                )
            else:
                merged.append(pending)
                pending = chunk

        if pending is not None:
            merged.append(pending)
        return merged

    def _can_merge_content_chunk(self, chunk: LegalChunk) -> bool:
        return chunk.source_node_type == "content" and not chunk.article_number


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    return len(_get_encoding(encoding_name).encode(text or ""))


def split_text(text: str, chunk_size: int, overlap: int, encoding_name: str = "cl100k_base") -> list[str]:
    splitter = _get_text_splitter(encoding_name, chunk_size, _effective_overlap(chunk_size, overlap))
    return [chunk for chunk in splitter.split_text(text) if _clean_text(chunk)]


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str):
    return tiktoken.get_encoding(encoding_name)


@lru_cache(maxsize=32)
def _get_text_splitter(encoding_name: str, chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )


def _effective_overlap(chunk_size: int, overlap: int) -> int:
    if chunk_size <= 1:
        return 0
    return min(max(0, overlap), max(0, chunk_size // 4))


def _merge_subtree_text(node: dict[str, Any]) -> str:
    parts = [_clean_text(node.get("text") or "")]
    for child in node.get("children") or []:
        child_text = _merge_subtree_text(child)
        if child_text:
            parts.append(child_text)
    return "\n".join(part for part in parts if part).strip()


def _node_label(node: dict[str, Any]) -> str:
    node_type = node.get("type") or "content"
    number = _clean_optional(node.get("number"))
    title = _clean_optional(node.get("title"))
    text = _clean_text(node.get("text") or "")

    if node_type == "article" and number:
        return f"\u0110i\u1ec1u {number}" + (f". {title}" if title else "")
    if node_type == "clause" and number:
        return f"Kho\u1ea3n {number}"
    if node_type == "point" and number:
        return f"\u0110i\u1ec3m {number}"
    if node_type in {"part", "appendix", "chapter", "section"}:
        return text.splitlines()[0] if text else f"{node_type} {number or ''}".strip()
    if node_type == "table":
        return "B\u1ea3ng"
    return ""
def _structural_heading_with_title_child(
    node: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    node_type = node.get("type") or "content"
    node_text = _clean_text(node.get("text") or "")
    children = node.get("children") or []

    if node_type not in CONTAINER_TYPES:
        return node_text, None

    if not children:
        return node_text, None

    first_child = children[0]
    first_type = first_child.get("type") or "content"
    first_text = _clean_text(first_child.get("text") or "")

    if first_type == "content" and _looks_like_title_line(first_text):
        heading = "\n".join(part for part in [node_text, first_text] if part).strip()
        return heading, first_child

    return node_text, None


def _looks_like_title_line(text: str) -> bool:
    text = _clean_text(text)
    if not text:
        return False

    first_line = text.splitlines()[0].strip()

    # Không coi khoản/điểm/điều là title-child.
    if re.match(r"(?i)^(điều\s+\d+|khoản\s+\d+|\d+\.|[a-zđ]\))\b", first_line):
        return False

    # Heading pháp luật thường ngắn, viết hoa nhiều, hoặc là cụm tiêu đề.
    if len(first_line) <= 120 and _uppercase_ratio(first_line) >= 0.55:
        return True

    return bool(
        re.search(
            r"(?i)\b(quy định chung|mẫu biểu|danh mục|quy cách|điều khoản thi hành|tổ chức thực hiện)\b",
            first_line,
        )
    )


def _uppercase_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zÀ-ỹĐđ]", text)
    if not letters:
        return 0.0

    uppercase = [ch for ch in letters if ch.upper() == ch and ch.lower() != ch]
    return len(uppercase) / len(letters)


def _is_rich_heading(text: str) -> bool:
    text = _clean_text(text)
    if not text:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # "Chương II" đơn độc => không chunk riêng.
    if len(lines) == 1 and re.match(r"(?i)^(chương|mục|phần)\s+[ivxlcdm\d]+$", lines[0]):
        return False

    # "Chương II\nMẪU BIỂU..." => chunk được.
    return True

def _document_context(metadata: dict[str, Any]) -> str:
    law_type = _clean_optional(metadata.get("loai_van_ban"))
    law_code = _clean_optional(metadata.get("so_ky_hieu"))
    title = _clean_optional(metadata.get("title"))
    return " ".join(value for value in [law_type, law_code, title] if value)


def _chunk_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "title",
        "so_ky_hieu",
        "ngay_ban_hanh",
        "loai_van_ban",
        "ngay_co_hieu_luc",
        "ngay_het_hieu_luc",
        "nganh",
        "linh_vuc",
        "co_quan_ban_hanh",
        "pham_vi",
        "tinh_trang_hieu_luc",
    ]
    return {key: metadata.get(key) for key in keys if key in metadata}


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text or "").strip())


def _is_decorative_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if not re.search(r"[\w\u00c0-\u1ef9]", compact, flags=re.UNICODE):
        return True
    return bool(re.fullmatch(r"[_\-–—=.*•|/\\(){}\[\]:;,.]+", compact))


def _looks_like_legal_heading(text: str) -> bool:
    first_line = _clean_text(text).splitlines()[0] if text else ""
    return bool(
        re.search(
            r"(?i)\b(điều|chương|mục|phần|phụ\s+lục|quyết\s+định|nghị\s+định|thông\s+tư|luật)\b",
            first_line,
        )
    )


def _is_markdown_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line.strip()))


def _replace_context_text(context_text: str, old_text: str, new_text: str) -> str:
    if context_text.endswith(old_text):
        return f"{context_text[: -len(old_text)]}{new_text}".strip()
    lines = [line for line in context_text.splitlines()[:-1] if line]
    return "\n".join([*lines, new_text]).strip()


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text
