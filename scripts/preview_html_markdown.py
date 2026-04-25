from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.chunking.html_to_markdown import html_to_markdown  # noqa: E402


DEFAULT_DOC_IDS = [186451, 186433, 186415, 186418, 186450]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Preview HTML to Markdown conversion for legal content.")
    parser.add_argument("--input", default="data/content_preview.csv", help="CSV file with id,content_html columns.")
    parser.add_argument("--doc-id", action="append", type=int, dest="doc_ids", help="Document id to preview.")
    parser.add_argument("--limit", type=int, default=1400, help="Max characters printed per document.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    doc_ids = args.doc_ids or DEFAULT_DOC_IDS

    for doc_id in doc_ids:
        matches = df[df["id"].eq(doc_id)]
        if matches.empty:
            print(f"\n--- doc_id={doc_id} not found ---")
            continue

        html = str(matches.iloc[0]["content_html"])
        markdown = html_to_markdown(html)
        html_tags_left = bool(re.search(r"</?[a-z][^>]*>", markdown, flags=re.IGNORECASE))
        table_count = markdown.count("| ---")

        print(f"\n--- doc_id={doc_id} chars={len(markdown)} markdown_tables={table_count} html_tags_left={html_tags_left} ---")
        print(markdown[: args.limit])


if __name__ == "__main__":
    main()
