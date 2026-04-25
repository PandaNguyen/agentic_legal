from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class SQLiteCheckpointStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ingest_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                args_json TEXT
            );

            CREATE TABLE IF NOT EXISTS doc_status (
                doc_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL,
                run_id TEXT
            );

            CREATE TABLE IF NOT EXISTS token_vocab (
                term_id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS metadata (
                doc_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS relationships (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                other_doc_id TEXT,
                relationship TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_relationships_doc_id ON relationships(doc_id);
            """
        )
        self.conn.commit()

    def start_run(self, run_id: str, args: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO ingest_runs(run_id, started_at, finished_at, status, args_json)
            VALUES (?, ?, NULL, ?, ?)
            """,
            (run_id, utc_now(), "running", json.dumps(args, ensure_ascii=False)),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE ingest_runs SET status = ?, finished_at = ? WHERE run_id = ?",
            (status, utc_now(), run_id),
        )
        self.conn.commit()

    def recover_interrupted_docs(self) -> int:
        cursor = self.conn.execute(
            """
            UPDATE doc_status
            SET status = 'failed',
                last_error = COALESCE(last_error, 'Interrupted previous run'),
                updated_at = ?
            WHERE status = 'processing'
            """,
            (utc_now(),),
        )
        self.conn.commit()
        return int(cursor.rowcount or 0)

    def get_doc_status(self, doc_id: str) -> str | None:
        row = self.conn.execute("SELECT status FROM doc_status WHERE doc_id = ?", (doc_id,)).fetchone()
        return str(row["status"]) if row else None

    def mark_processing(self, doc_id: str, run_id: str) -> None:
        self.conn.execute(
            """
            INSERT INTO doc_status(doc_id, status, attempt_count, chunk_count, last_error, updated_at, run_id)
            VALUES (?, 'processing', 1, 0, NULL, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                status = 'processing',
                attempt_count = doc_status.attempt_count + 1,
                updated_at = excluded.updated_at,
                run_id = excluded.run_id
            """,
            (doc_id, utc_now(), run_id),
        )
        self.conn.commit()

    def mark_done(self, doc_id: str, run_id: str, chunk_count: int) -> None:
        self.conn.execute(
            """
            INSERT INTO doc_status(doc_id, status, attempt_count, chunk_count, last_error, updated_at, run_id)
            VALUES (?, 'done', 1, ?, NULL, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                status = 'done',
                chunk_count = excluded.chunk_count,
                last_error = NULL,
                updated_at = excluded.updated_at,
                run_id = excluded.run_id
            """,
            (doc_id, chunk_count, utc_now(), run_id),
        )
        self.conn.commit()

    def mark_failed(self, doc_id: str, run_id: str, error: str) -> None:
        self.conn.execute(
            """
            INSERT INTO doc_status(doc_id, status, attempt_count, chunk_count, last_error, updated_at, run_id)
            VALUES (?, 'failed', 1, 0, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                status = 'failed',
                last_error = excluded.last_error,
                updated_at = excluded.updated_at,
                run_id = excluded.run_id
            """,
            (doc_id, error, utc_now(), run_id),
        )
        self.conn.commit()

    def import_metadata_csv(self, csv_path: str | Path, force: bool = False, batch_size: int = 1000) -> int:
        if not force and self._table_has_rows("metadata"):
            return 0
        if force:
            self.conn.execute("DELETE FROM metadata")
            self.conn.commit()

        inserted = 0
        buffer: list[tuple[str, str]] = []
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                doc_id = str(row.get("id") or "").strip()
                if not doc_id:
                    continue
                payload = clean_record(row)
                buffer.append((doc_id, json.dumps(payload, ensure_ascii=False)))
                if len(buffer) >= batch_size:
                    inserted += self._flush_metadata_rows(buffer)
                    buffer.clear()
        if buffer:
            inserted += self._flush_metadata_rows(buffer)
        return inserted

    def import_relationships_csv(self, csv_path: str | Path, force: bool = False, batch_size: int = 2000) -> int:
        if not force and self._table_has_rows("relationships"):
            return 0
        if force:
            self.conn.execute("DELETE FROM relationships")
            self.conn.commit()

        inserted = 0
        buffer: list[tuple[str, str | None, str | None, str]] = []
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                doc_id = str(row.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                payload = clean_record(row)
                buffer.append(
                    (
                        doc_id,
                        clean_optional(payload.get("other_doc_id")),
                        clean_optional(payload.get("relationship")),
                        json.dumps(payload, ensure_ascii=False),
                    )
                )
                if len(buffer) >= batch_size:
                    inserted += self._flush_relationship_rows(buffer)
                    buffer.clear()
        if buffer:
            inserted += self._flush_relationship_rows(buffer)
        return inserted

    def get_metadata(self, doc_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT payload_json FROM metadata WHERE doc_id = ?", (doc_id,)).fetchone()
        return json.loads(row["payload_json"]) if row else {"id": doc_id}

    def get_relationships(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload_json FROM relationships WHERE doc_id = ? ORDER BY row_id ASC",
            (doc_id,),
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def resolve_term_ids(self, terms: Iterable[str], create_missing: bool = False) -> dict[str, int]:
        unique_terms = sorted({str(term).strip() for term in terms if str(term).strip()})
        if not unique_terms:
            return {}

        found = self._select_terms(unique_terms)
        missing = [term for term in unique_terms if term not in found]
        if create_missing and missing:
            self.conn.executemany("INSERT OR IGNORE INTO token_vocab(term) VALUES (?)", ((term,) for term in missing))
            self.conn.commit()
            found = self._select_terms(unique_terms)
        return found

    def _flush_metadata_rows(self, rows: list[tuple[str, str]]) -> int:
        self.conn.executemany(
            "INSERT OR REPLACE INTO metadata(doc_id, payload_json) VALUES (?, ?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def _flush_relationship_rows(self, rows: list[tuple[str, str | None, str | None, str]]) -> int:
        self.conn.executemany(
            """
            INSERT INTO relationships(doc_id, other_doc_id, relationship, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def _select_terms(self, terms: list[str]) -> dict[str, int]:
        placeholders = ",".join("?" for _ in terms)
        rows = self.conn.execute(
            f"SELECT term, term_id FROM token_vocab WHERE term IN ({placeholders})",
            terms,
        ).fetchall()
        return {str(row["term"]): int(row["term_id"]) for row in rows}

    def _table_has_rows(self, table_name: str) -> bool:
        row = self.conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1").fetchone()
        return row is not None


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: clean_optional(value) for key, value in record.items()}


def clean_optional(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
