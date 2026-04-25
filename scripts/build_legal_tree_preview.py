from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.chunking.html_to_markdown import html_to_markdown  # noqa: E402
from app.services.chunking.legal_tree_builder import (  # noqa: E402
    build_adjacency_list,
    build_closure_table,
    build_document_tree,
    flatten_tree_nodes,
)


DEFAULT_OUTPUT_DIR = Path("data/processed")
METADATA_FIELDS = [
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build legal document trees for preview content.")
    parser.add_argument("--content", default="data/content_preview.csv", help="CSV with id,content_html columns.")
    parser.add_argument("--metadata", default="data/metadata.csv", help="CSV with document metadata.")
    parser.add_argument("--relationships", default="data/relationships.csv", help="CSV with document relationships.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSONL artifacts.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of content rows to process.")
    args = parser.parse_args()

    content_df = pd.read_csv(args.content, encoding="utf-8")
    if args.limit:
        content_df = content_df.head(args.limit)
    metadata_df = pd.read_csv(args.metadata, encoding="utf-8")
    relationships_df = pd.read_csv(args.relationships, encoding="utf-8")

    metadata_by_id = {int(row["id"]): _clean_record(row.to_dict()) for _, row in metadata_df.iterrows()}
    content_doc_ids = {int(doc_id) for doc_id in content_df["id"].tolist()}
    relationships_by_doc = _build_relationship_index(relationships_df, metadata_by_id, content_doc_ids)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_path = output_dir / "legal_tree_preview.jsonl"
    nodes_path = output_dir / "legal_tree_nodes_preview.jsonl"
    closure_path = output_dir / "legal_tree_closure_preview.jsonl"
    adjacency_path = output_dir / "legal_tree_adjacency_preview.jsonl"

    totals = {
        "docs_processed": 0,
        "metadata_matches": 0,
        "empty_docs": 0,
        "trees_built": 0,
        "relationship_rows_attached": 0,
        "article_nodes": 0,
        "chapter_nodes": 0,
        "table_nodes": 0,
    }

    with (
        tree_path.open("w", encoding="utf-8") as tree_file,
        nodes_path.open("w", encoding="utf-8") as nodes_file,
        closure_path.open("w", encoding="utf-8") as closure_file,
        adjacency_path.open("w", encoding="utf-8") as adjacency_file,
    ):
        for _, row in content_df.iterrows():
            doc_id_int = int(row["id"])
            doc_id = str(doc_id_int)
            metadata = metadata_by_id.get(doc_id_int, {"id": doc_id_int})
            relationships = relationships_by_doc.get(doc_id_int, [])
            markdown = html_to_markdown(row.get("content_html"))

            document_tree = build_document_tree(
                doc_id=doc_id,
                markdown=markdown,
                metadata=metadata,
                relationships=relationships,
            )
            flat_nodes = flatten_tree_nodes(document_tree.tree)
            closure_edges = build_closure_table(document_tree.tree)
            adjacency_edges = build_adjacency_list(document_tree.tree)

            _write_jsonl(tree_file, document_tree.model_dump(mode="json"))
            for node in flat_nodes:
                _write_jsonl(nodes_file, node.model_dump(mode="json", exclude={"children"}))
            for edge in closure_edges:
                _write_jsonl(closure_file, edge.model_dump(mode="json"))
            for edge in adjacency_edges:
                _write_jsonl(adjacency_file, edge.model_dump(mode="json"))

            totals["docs_processed"] += 1
            totals["metadata_matches"] += int(doc_id_int in metadata_by_id)
            totals["empty_docs"] += int(document_tree.parse_status == "empty_content")
            totals["trees_built"] += 1
            totals["relationship_rows_attached"] += len(relationships)
            totals["article_nodes"] += document_tree.stats.get("article", 0)
            totals["chapter_nodes"] += document_tree.stats.get("chapter", 0)
            totals["table_nodes"] += document_tree.stats.get("table", 0)

    print("Legal tree preview build complete")
    for key, value in totals.items():
        print(f"{key}: {value}")
    print(f"tree: {tree_path}")
    print(f"nodes: {nodes_path}")
    print(f"closure: {closure_path}")
    print(f"adjacency: {adjacency_path}")


def _build_relationship_index(
    relationships_df: pd.DataFrame,
    metadata_by_id: dict[int, dict[str, Any]],
    doc_ids: set[int],
) -> dict[int, list[dict[str, Any]]]:
    relationship_index: dict[int, list[dict[str, Any]]] = {}
    scoped_relationships = relationships_df[relationships_df["doc_id"].isin(doc_ids)]
    for row in scoped_relationships.itertuples(index=False):
        record = _clean_record(row._asdict())
        doc_id = int(record["doc_id"])
        other_doc_id = int(record["other_doc_id"])
        target_metadata = metadata_by_id.get(other_doc_id)
        if target_metadata:
            record["target_metadata"] = {field: target_metadata.get(field) for field in METADATA_FIELDS if field in target_metadata}
        relationship_index.setdefault(doc_id, []).append(record)
    return relationship_index


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean_value(value) for key, value in record.items()}


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def _write_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
