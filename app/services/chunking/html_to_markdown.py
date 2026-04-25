from __future__ import annotations

import re
from html import unescape

from bs4 import BeautifulSoup, NavigableString, Tag


BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "div",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "p",
    "section",
}
LEGAL_HEADER_PATTERNS = (
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "Độc lập - Tự do - Hạnh phúc",
    "Độc lập – Tự do – Hạnh phúc",
    "ỦY BAN NHÂN DÂN",
    "UỶ BAN NHÂN DÂN",
    "QUỐC HỘI",
    "CHÍNH PHỦ",
)


def html_to_markdown(html: str | None) -> str:
    """Convert raw legal HTML into retrieval-friendly Markdown.

    VBPL pages often wrap the whole document in layout tables. Those tables are
    unwrapped, while dense data tables are preserved as Markdown tables.
    """
    if not html or not str(html).strip():
        return ""

    soup = BeautifulSoup(str(html), "html.parser")
    for tag in soup(["script", "style", "img", "noscript"]):
        tag.decompose()

    root = soup.find(id="content") or soup.body or soup
    markdown = _render_children(root)
    return _normalize_markdown(markdown)


def _render_node(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return _normalize_inline(str(node), strip_edges=False)

    if not isinstance(node, Tag):
        return ""

    name = node.name.lower()
    if name == "br":
        return "\n"
    if name == "table":
        return _render_table(node)
    if name in {"thead", "tbody", "tfoot"}:
        return _render_children(node)
    if name == "tr":
        return _render_children(node)
    if name in {"td", "th"}:
        return _render_children(node)
    if name in {"strong", "b", "em", "i", "span", "a", "u"}:
        return _render_inline_children(node)
    if name in BLOCK_TAGS:
        content = _render_children(node)
        return _as_block(content)

    return _render_children(node)


def _render_children(tag: Tag) -> str:
    parts = [_render_node(child) for child in tag.children]
    return "".join(parts)


def _render_inline_children(tag: Tag) -> str:
    text = "".join(_render_node(child) for child in tag.children)
    return _normalize_inline(text, strip_edges=False)


def _render_table(table: Tag) -> str:
    rows = _extract_table_rows(table)
    if _is_layout_table(table, rows):
        return _as_block(_render_children_without_table_shell(table))

    return _as_block(_rows_to_markdown(rows))


def _render_children_without_table_shell(table: Tag) -> str:
    parts: list[str] = []
    for row in _direct_rows(table):
        cells = row.find_all(["td", "th"], recursive=False)
        rendered_cells = [_render_children(cell) for cell in cells]
        parts.append("\n\n".join(cell for cell in rendered_cells if cell.strip()))
    return "\n\n".join(part for part in parts if part.strip())


def _direct_rows(table: Tag) -> list[Tag]:
    rows: list[Tag] = []
    for child in table.children:
        if not isinstance(child, Tag):
            continue
        if child.name and child.name.lower() == "tr":
            rows.append(child)
        elif child.name and child.name.lower() in {"thead", "tbody", "tfoot"}:
            rows.extend(child.find_all("tr", recursive=False))
    return rows


def _extract_table_rows(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in _direct_rows(table):
        row: list[str] = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            colspan = _safe_int(cell.get("colspan"), default=1)
            text = _cell_text(cell)
            row.append(text)
            for _ in range(max(0, colspan - 1)):
                row.append("")
        if any(value.strip() for value in row):
            rows.append(row)

    if not rows:
        return []

    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _cell_text(cell: Tag) -> str:
    lines = [_normalize_inline(line) for line in cell.get_text(separator="\n", strip=True).splitlines()]
    text = " / ".join(line for line in lines if line)
    return _escape_table_cell(text)


def _is_layout_table(table: Tag, rows: list[list[str]]) -> bool:
    if _has_class(table, "detailcontent"):
        return True
    if not rows:
        return True

    row_count = len(rows)
    col_count = max(len(row) for row in rows)
    non_empty_cells = [cell for row in rows for cell in row if cell.strip()]
    joined = _strip_accents_for_match(" ".join(non_empty_cells))

    if len(non_empty_cells) <= 1:
        return True
    if row_count <= 1:
        return True
    if col_count <= 1:
        return True
    if _looks_like_legal_header(joined, row_count):
        return True
    if _looks_like_decoration_table(table, non_empty_cells):
        return True

    return False


def _looks_like_legal_header(joined_upper_text: str, row_count: int) -> bool:
    normalized_patterns = [_strip_accents_for_match(pattern) for pattern in LEGAL_HEADER_PATTERNS]
    hits = sum(1 for pattern in normalized_patterns if pattern in joined_upper_text)
    has_document_number = bool(re.search(r"\bSO\s*[:：]", joined_upper_text))
    has_date_line = "NGAY" in joined_upper_text and "THANG" in joined_upper_text and "NAM" in joined_upper_text
    return row_count <= 4 and (hits >= 1 or (has_document_number and has_date_line))


def _looks_like_decoration_table(table: Tag, non_empty_cells: list[str]) -> bool:
    if table.find("img"):
        return True
    text = "".join(non_empty_cells).strip("_-–— ")
    return not text


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    header = rows[0]
    body = rows[1:] or [[""] * len(header)]
    separator = ["---"] * len(header)
    markdown_rows = [_format_table_row(header), _format_table_row(separator)]
    markdown_rows.extend(_format_table_row(row) for row in body)
    return "\n".join(markdown_rows)


def _format_table_row(row: list[str]) -> str:
    return "| " + " | ".join(cell.strip() for cell in row) + " |"


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _as_block(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return f"\n\n{text}\n\n"


def _normalize_inline(text: str, strip_edges: bool = False) -> str:
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip(" ") if strip_edges else text


def _normalize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\xa0", " ")
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n[ \t]+", "\n", markdown)
    markdown = _fix_missing_boundary_spaces(markdown)
    markdown = _sanitize_residual_html_markup(markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"[ \t]{2,}", " ", markdown)
    return markdown.strip()


def _fix_missing_boundary_spaces(text: str) -> str:
    if not text:
        return ""

    output: list[str] = [text[0]]
    for previous, current in zip(text, text[1:]):
        needs_space = (previous.islower() and (current.isupper() or current.isdigit())) or (
            previous.isdigit() and current.isalpha()
        )
        if needs_space:
            output.append(" ")
        output.append(current)
    return "".join(output)


def _sanitize_residual_html_markup(text: str) -> str:
    """Remove malformed tag fragments that come from escaped legal table text.

    Some source cells contain comparisons such as ``30%&lt;Fe≤40%`` near broken
    paragraph tags. BeautifulSoup can interpret those as bogus tags, so this
    pass keeps the comparison readable and removes the leftover markup.
    """
    text = re.sub(r"<\s*/?\s*p(?:\s+[^>]*)?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</[^>\s]+>", " ", text)
    text = re.sub(r"(?<=\d%)<(?=[A-Za-zÀ-ỹ])", " < ", text)
    text = re.sub(r"(?<=[A-Za-zÀ-ỹ])<(?=\d)", " < ", text)
    text = re.sub(r"</?[a-z][^>]*>", " ", text, flags=re.IGNORECASE)
    return text


def _has_class(tag: Tag, class_name: str) -> bool:
    classes = tag.get("class") or []
    return any(str(value).lower() == class_name.lower() for value in classes)


def _safe_int(value: object, default: int) -> int:
    try:
        return max(1, int(str(value)))
    except (TypeError, ValueError):
        return default


def _strip_accents_for_match(text: str) -> str:
    replacements = str.maketrans(
        {
            "À": "A",
            "Á": "A",
            "Ả": "A",
            "Ã": "A",
            "Ạ": "A",
            "Ă": "A",
            "Ằ": "A",
            "Ắ": "A",
            "Ẳ": "A",
            "Ẵ": "A",
            "Ặ": "A",
            "Â": "A",
            "Ầ": "A",
            "Ấ": "A",
            "Ẩ": "A",
            "Ẫ": "A",
            "Ậ": "A",
            "Đ": "D",
            "È": "E",
            "É": "E",
            "Ẻ": "E",
            "Ẽ": "E",
            "Ẹ": "E",
            "Ê": "E",
            "Ề": "E",
            "Ế": "E",
            "Ể": "E",
            "Ễ": "E",
            "Ệ": "E",
            "Ì": "I",
            "Í": "I",
            "Ỉ": "I",
            "Ĩ": "I",
            "Ị": "I",
            "Ò": "O",
            "Ó": "O",
            "Ỏ": "O",
            "Õ": "O",
            "Ọ": "O",
            "Ô": "O",
            "Ồ": "O",
            "Ố": "O",
            "Ổ": "O",
            "Ỗ": "O",
            "Ộ": "O",
            "Ơ": "O",
            "Ờ": "O",
            "Ớ": "O",
            "Ở": "O",
            "Ỡ": "O",
            "Ợ": "O",
            "Ù": "U",
            "Ú": "U",
            "Ủ": "U",
            "Ũ": "U",
            "Ụ": "U",
            "Ư": "U",
            "Ừ": "U",
            "Ứ": "U",
            "Ử": "U",
            "Ữ": "U",
            "Ự": "U",
            "Ỳ": "Y",
            "Ý": "Y",
            "Ỷ": "Y",
            "Ỹ": "Y",
            "Ỵ": "Y",
        }
    )
    return text.upper().translate(replacements)
