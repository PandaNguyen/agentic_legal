from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from app.services.chunking.legal_tree_models import AdjacencyEdge, ClosureEdge, LegalDocumentTree, LegalTreeNode


DIEU = "Điều"
CHUONG = "Chương"
MUC = "Mục"
PHAN = "Phần"
PHU_LUC = "Phụ lục"

LEVELS = {
    "document": 0.0,
    "part": 1.0,
    "appendix": 1.0,
    "chapter": 2.0,
    "section": 3.0,
    "article": 4.0,
    "clause": 5.0,
    "point": 6.0,
    "content": 7.0,
    "table": 7.0,
}

EDIT_ARTICLE_PATTERN = re.compile(
    rf"(?i)^({DIEU}\s+\d+\s*[.)-]\s*)?"
    r"(sửa đổi|bổ sung|thay đổi|thay thế|"
    r"b\u00e3i b\u1ecf|h\u1ee7y b\u1ecf|hu\u1ef7 b\u1ecf|"
    r"ch\u1ec9nh s\u1eeda|\u0111\u00ecnh ch\u1ec9)"
)


def markdown_to_blocks(markdown: str) -> list[str]:
    """Split Markdown into paragraph-like blocks while keeping tables intact."""
    if not markdown:
        return []

    blocks: list[str] = []
    current_table: list[str] = []
    current_text: list[str] = []

    def flush_text() -> None:
        if current_text:
            block = "\n".join(current_text).strip()
            if block:
                blocks.append(block)
            current_text.clear()

    def flush_table() -> None:
        if current_table:
            block = "\n".join(current_table).strip()
            if block:
                blocks.append(block)
            current_table.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if _is_markdown_table_line(line):
            flush_text()
            current_table.append(line)
            continue

        flush_table()
        if not line:
            flush_text()
            continue
        current_text.append(line)

    flush_table()
    flush_text()
    return blocks


def detect_level(block: str) -> dict[str, Any]:
    text = _normalize_text(block)
    first_line = text.splitlines()[0].strip() if text else ""

    if _is_markdown_table(text):
        return {"type": "table", "level": LEVELS["table"], "number": None, "title": None}

    patterns: list[tuple[str, float, re.Pattern[str]]] = [
        ("appendix", LEVELS["appendix"], re.compile(rf"^{PHU_LUC}\s*([IVXLCDM]+|\d+)?(?:[.)-])?", re.I)),
        ("part", LEVELS["part"], re.compile(rf"^{PHAN}\s+([IVXLCDM]+|\d+)(?:[.)-])?", re.I)),
        ("chapter", LEVELS["chapter"], re.compile(rf"^{CHUONG}\s+([IVXLCDM]+|\d+)(?:[.)-])?", re.I)),
        ("section", LEVELS["section"], re.compile(rf"^{MUC}\s+([IVXLCDM]+|\d+)(?:[.)-])?", re.I)),
        ("article", LEVELS["article"], re.compile(rf"^{DIEU}\s+(\d+[a-zA-Z]?)\s*[.)-]?\s*(.*)", re.I)),
        ("clause", LEVELS["clause"], re.compile(r"^(\d+)\s*[.)-]\s+(.+)", re.I)),
        ("point", LEVELS["point"], re.compile(r"^([a-z\u0111])\s*[.)-]\s+(.+)", re.I)),
    ]
    for node_type, level, pattern in patterns:
        match = pattern.search(first_line)
        if match:
            number = match.group(1) or None
            title = match.group(2).strip() if node_type in {"article", "clause", "point"} and match.lastindex and match.lastindex >= 2 else None
            return {"type": node_type, "level": level, "number": number, "title": title or None}

    return {"type": "content", "level": LEVELS["content"], "number": None, "title": None}


def build_nodes(doc_id: str, markdown: str, metadata: dict[str, Any] | None = None) -> list[LegalTreeNode]:
    blocks = markdown_to_blocks(markdown)
    nodes: list[LegalTreeNode] = []

    for index, block in enumerate(blocks, start=1):
        detected = detect_level(block)
        node_type = detected["type"]
        number = detected["number"]
        nodes.append(
            LegalTreeNode(
                node_id=f"{doc_id}:block:{index}",
                doc_id=doc_id,
                type=node_type,
                number=number,
                title=detected["title"],
                text=block,
                raw_text=block,
                level=detected["level"],
                path=[],
                metadata={"block_index": index, **(metadata or {})},
                is_edit=_is_edit_block(block),
            )
        )

    return nodes


def build_document_tree(
    doc_id: str,
    markdown: str,
    metadata: dict[str, Any] | None = None,
    relationships: list[dict[str, Any]] | None = None,
) -> LegalDocumentTree:
    doc_metadata = metadata or {}
    root = LegalTreeNode(
        node_id=f"{doc_id}:document",
        doc_id=doc_id,
        type="document",
        number=None,
        title=_clean_optional(doc_metadata.get("title")),
        text=_clean_optional(doc_metadata.get("title")) or "",
        raw_text=_clean_optional(doc_metadata.get("title")) or "",
        level=LEVELS["document"],
        parent_id=None,
        path=[f"document:{doc_id}"],
        metadata=doc_metadata,
    )
    nodes = build_nodes(doc_id=doc_id, markdown=markdown, metadata=doc_metadata)
    tree = build_tree(root=root, nodes=nodes)
    stats = _collect_stats(tree)
    parse_status = "ok" if markdown.strip() else "empty_content"
    if markdown.strip() and not nodes:
        parse_status = "no_blocks"

    return LegalDocumentTree(
        doc_id=doc_id,
        metadata=doc_metadata,
        relationships=relationships or [],
        parse_status=parse_status,
        tree=tree,
        stats=stats,
    )


def build_tree(root: LegalTreeNode, nodes: list[LegalTreeNode]) -> LegalTreeNode:
    stack: list[LegalTreeNode] = [root]
    sibling_counts: dict[tuple[str, str], int] = defaultdict(int)

    for node in nodes:
        while len(stack) > 1 and stack[-1].level >= node.level:
            stack.pop()

        parent = stack[-1]
        node.parent_id = parent.node_id
        segment = _path_segment(node)
        sibling_key = (parent.node_id, segment)
        sibling_counts[sibling_key] += 1
        if sibling_counts[sibling_key] > 1:
            segment = f"{segment}:{sibling_counts[sibling_key]}"
        node.node_id = f"{parent.node_id}:{segment}"
        node.path = [*parent.path, segment]
        parent.children.append(node)
        stack.append(node)

    return root


def build_closure_table(tree: LegalTreeNode) -> list[ClosureEdge]:
    edges: list[ClosureEdge] = []

    def traverse(node: LegalTreeNode, ancestors: list[str]) -> None:
        for depth, ancestor_id in enumerate(reversed(ancestors), start=1):
            edges.append(ClosureEdge(doc_id=node.doc_id, ancestor=ancestor_id, descendant=node.node_id, depth=depth))
        edges.append(ClosureEdge(doc_id=node.doc_id, ancestor=node.node_id, descendant=node.node_id, depth=0))
        for child in node.children:
            traverse(child, [*ancestors, node.node_id])

    traverse(tree, [])
    return edges


def build_adjacency_list(tree: LegalTreeNode) -> list[AdjacencyEdge]:
    edges: list[AdjacencyEdge] = []

    def traverse(node: LegalTreeNode) -> None:
        edges.append(AdjacencyEdge(doc_id=node.doc_id, parent=node.node_id, children=[child.node_id for child in node.children]))
        for child in node.children:
            traverse(child)

    traverse(tree)
    return edges


def flatten_tree_nodes(tree: LegalTreeNode) -> list[LegalTreeNode]:
    nodes: list[LegalTreeNode] = []

    def traverse(node: LegalTreeNode) -> None:
        nodes.append(node)
        for child in node.children:
            traverse(child)

    traverse(tree)
    return nodes


def _collect_stats(tree: LegalTreeNode) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for node in flatten_tree_nodes(tree):
        counts[node.type] += 1
    return dict(counts)


def _is_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) >= 2 and _is_markdown_table_line(lines[0]) and any(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line) for line in lines[1:3])


def _is_markdown_table_line(line: str) -> bool:
    return line.startswith("|") and line.endswith("|")


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _path_segment(node: LegalTreeNode) -> str:
    value = _slugify(node.number or node.title or str(node.metadata.get("block_index", "")) or node.node_id)
    return f"{node.type}:{value}"


def _slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "none"


def _is_edit_block(text: str) -> bool:
    return bool(EDIT_ARTICLE_PATTERN.search(_normalize_text(text)))


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text
