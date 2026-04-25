from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.chunking.legal_chunk_extractor import ChunkingConfig, LegalTreeChunkExtractor  # noqa: E402


DEFAULT_TREE_INPUT = Path("data/processed/legal_tree_preview.jsonl")
DEFAULT_CHUNK_OUTPUT = Path("data/processed/legal_chunks_previewv2.jsonl")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build retrieval chunks from legal tree preview JSONL.")
    parser.add_argument("--input", default=str(DEFAULT_TREE_INPUT), help="Legal tree JSONL input.")
    parser.add_argument("--output", default=str(DEFAULT_CHUNK_OUTPUT), help="Legal chunks JSONL output.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Approximate max token count per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Approximate overlap for long text chunks.")
    parser.add_argument("--table-chunk-size", type=int, default=500, help="Approximate max token count per table chunk.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of documents to process.")
    args = parser.parse_args()

    extractor = LegalTreeChunkExtractor(
        ChunkingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            table_chunk_size=args.table_chunk_size,
        )
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "docs_processed": 0,
        "docs_with_chunks": 0,
        "chunks_written": 0,
        "article_chunks": 0,
        "table_chunks": 0,
        "max_chunk_tokens": 0,
    }

    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if args.limit is not None and totals["docs_processed"] >= args.limit:
                break
            document_tree = json.loads(line)
            chunks = extractor.extract_document_chunks(document_tree)
            totals["docs_processed"] += 1
            totals["docs_with_chunks"] += int(bool(chunks))

            for chunk in chunks:
                record = chunk.model_dump(mode="json")
                _write_jsonl(target, record)
                totals["chunks_written"] += 1
                totals["article_chunks"] += int(bool(chunk.article_number))
                totals["table_chunks"] += int(chunk.source_node_type == "table")
                totals["max_chunk_tokens"] = max(totals["max_chunk_tokens"], chunk.token_count)

    print("Legal chunk preview build complete")
    for key, value in totals.items():
        print(f"{key}: {value}")
    print(f"chunks: {output_path}")


def _write_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
