from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from medical_kg.openalex.models import OpenAlexWork, normalize_work_id


class OpenAlexCatalog:
    """Small, maintainable catalog of screened Works and their stable locators."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS works (
                work_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL UNIQUE,
                openalex_id TEXT NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT,
                doi TEXT,
                publication_year INTEGER,
                language TEXT,
                work_type TEXT,
                primary_source_id TEXT,
                primary_source_name TEXT,
                sources_json TEXT NOT NULL,
                has_fulltext_hint INTEGER NOT NULL,
                fulltext_urls_json TEXT NOT NULL,
                snapshot_file TEXT NOT NULL,
                snapshot_line INTEGER NOT NULL,
                raw_json TEXT NOT NULL,
                selected INTEGER NOT NULL,
                selection_method TEXT NOT NULL,
                matched_keywords_json TEXT NOT NULL,
                fulltext_path TEXT,
                fulltext_status TEXT NOT NULL DEFAULT 'unresolved',
                materialized_path TEXT,
                abstract_processed INTEGER NOT NULL DEFAULT 0,
                abstract_materialized_path TEXT,
                fulltext_processed INTEGER NOT NULL DEFAULT 0,
                fulltext_materialized_path TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS ix_openalex_works_selected ON works(selected);
            CREATE INDEX IF NOT EXISTS ix_openalex_works_doi ON works(doi);
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                display_name TEXT,
                raw_json TEXT NOT NULL,
                is_full_record INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS selection_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                options_json TEXT NOT NULL,
                scanned INTEGER NOT NULL DEFAULT 0,
                candidates INTEGER NOT NULL DEFAULT 0,
                selected INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        # CREATE TABLE IF NOT EXISTS does not add columns to catalogs created by
        # earlier releases. Keep the catalog forward-compatible in place.
        existing_columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(works)")
        }
        migrations = {
            "abstract_processed": "INTEGER NOT NULL DEFAULT 0",
            "abstract_materialized_path": "TEXT",
            "fulltext_processed": "INTEGER NOT NULL DEFAULT 0",
            "fulltext_materialized_path": "TEXT",
        }
        for name, definition in migrations.items():
            if name not in existing_columns:
                self.connection.execute(f"ALTER TABLE works ADD COLUMN {name} {definition}")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> OpenAlexCatalog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start_run(self, options: dict[str, Any]) -> int:
        cursor = self.connection.execute(
            "INSERT INTO selection_runs(options_json) VALUES (?)",
            (json.dumps(options, ensure_ascii=False),),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, *, scanned: int, candidates: int, selected: int) -> None:
        self.connection.execute(
            "UPDATE selection_runs SET scanned=?, candidates=?, selected=? WHERE run_id=?",
            (scanned, candidates, selected, run_id),
        )
        self.connection.commit()

    def upsert_work(
        self,
        work: OpenAlexWork,
        *,
        selected: bool,
        selection_method: str,
        matched_keywords: list[str] | None = None,
        commit: bool = True,
    ) -> None:
        values = (
            work.work_id,
            work.document_id,
            work.openalex_id,
            work.title,
            work.abstract,
            work.doi,
            work.publication_year,
            work.language,
            work.work_type,
            work.primary_source_id,
            work.primary_source_name,
            json.dumps(work.sources, ensure_ascii=False),
            int(work.has_fulltext_hint),
            json.dumps(work.fulltext_urls, ensure_ascii=False),
            work.snapshot_file,
            work.snapshot_line,
            json.dumps(work.raw, ensure_ascii=False),
            int(selected),
            selection_method,
            json.dumps(matched_keywords or [], ensure_ascii=False),
        )
        self.connection.execute(
            """
            INSERT INTO works (
                work_id, document_id, openalex_id, title, abstract, doi,
                publication_year, language, work_type, primary_source_id,
                primary_source_name, sources_json, has_fulltext_hint,
                fulltext_urls_json, snapshot_file, snapshot_line, raw_json,
                selected, selection_method, matched_keywords_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(work_id) DO UPDATE SET
                title=excluded.title, abstract=excluded.abstract, doi=excluded.doi,
                publication_year=excluded.publication_year, language=excluded.language,
                work_type=excluded.work_type, primary_source_id=excluded.primary_source_id,
                primary_source_name=excluded.primary_source_name,
                sources_json=excluded.sources_json,
                has_fulltext_hint=excluded.has_fulltext_hint,
                fulltext_urls_json=excluded.fulltext_urls_json,
                snapshot_file=excluded.snapshot_file, snapshot_line=excluded.snapshot_line,
                raw_json=excluded.raw_json,
                selected=CASE WHEN works.selection_method='manual' THEN 1
                              ELSE excluded.selected END,
                selection_method=CASE WHEN works.selection_method='manual' THEN 'manual'
                                      ELSE excluded.selection_method END,
                matched_keywords_json=excluded.matched_keywords_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            values,
        )
        self.upsert_sources(work.sources, full=False)
        if commit:
            self.connection.commit()

    def upsert_sources(self, sources: Iterable[dict[str, Any]], *, full: bool) -> None:
        for source in sources:
            source_id = source.get("id")
            if not source_id:
                continue
            self.connection.execute(
                """
                INSERT INTO sources(source_id, display_name, raw_json, is_full_record)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    display_name=COALESCE(excluded.display_name, sources.display_name),
                    raw_json=CASE WHEN excluded.is_full_record >= sources.is_full_record
                                  THEN excluded.raw_json ELSE sources.raw_json END,
                    is_full_record=MAX(sources.is_full_record, excluded.is_full_record),
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    str(source_id),
                    source.get("display_name"),
                    json.dumps(source, ensure_ascii=False),
                    int(full),
                ),
            )

    def selected_rows(self, *, limit: int | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM works WHERE selected=1 ORDER BY work_id"
        parameters: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            parameters = (limit,)
        return list(self.connection.execute(sql, parameters))

    def get(self, work_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM works WHERE work_id=?", (normalize_work_id(work_id),)
        ).fetchone()

    def selected_source_ids(self) -> set[str]:
        result: set[str] = set()
        for row in self.selected_rows():
            for source in json.loads(row["sources_json"]):
                if source.get("id"):
                    result.add(str(source["id"]))
        return result

    def selected_materialized_paths(
        self, *, content_mode: str = "fulltext", limit: int | None = None
    ) -> list[Path]:
        if content_mode == "fulltext":
            columns = ("fulltext_materialized_path",)
        elif content_mode == "abstract":
            columns = ("abstract_materialized_path",)
        elif content_mode == "fulltext-or-abstract":
            columns = ("fulltext_materialized_path", "abstract_materialized_path")
        else:
            raise ValueError("Invalid content_mode")
        expressions = [
            f"SELECT {column} AS path FROM works WHERE selected=1 AND {column} IS NOT NULL"
            for column in columns
        ]
        sql = "SELECT DISTINCT path FROM (" + " UNION ALL ".join(expressions) + ") ORDER BY path"
        parameters: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            parameters = (limit,)
        return [
            Path(str(row[0]))
            for row in self.connection.execute(sql, parameters)
            if Path(str(row[0])).is_file()
        ]

    def set_fulltext_materialized(
        self,
        work_id: str,
        *,
        fulltext_path: str | None,
        fulltext_status: str,
        materialized_path: str | None,
        processed: bool,
    ) -> None:
        self.connection.execute(
            """UPDATE works SET fulltext_path=?, fulltext_status=?, materialized_path=?,
               fulltext_materialized_path=?, fulltext_processed=?,
               updated_at=CURRENT_TIMESTAMP WHERE work_id=?""",
            (
                fulltext_path,
                fulltext_status,
                materialized_path,
                materialized_path,
                int(processed),
                normalize_work_id(work_id),
            ),
        )
        self.connection.commit()

    def set_materialized(
        self,
        work_id: str,
        *,
        fulltext_path: str | None,
        fulltext_status: str,
        materialized_path: str | None,
    ) -> None:
        """Backward-compatible alias for full-text materialization records."""

        self.set_fulltext_materialized(
            work_id,
            fulltext_path=fulltext_path,
            fulltext_status=fulltext_status,
            materialized_path=materialized_path,
            processed=materialized_path is not None,
        )

    def set_abstract_materialized(self, work_ids: Iterable[str], *, materialized_path: str) -> None:
        normalized = [(materialized_path, normalize_work_id(work_id)) for work_id in work_ids]
        self.connection.executemany(
            """UPDATE works SET abstract_materialized_path=?, abstract_processed=1,
               materialized_path=?, updated_at=CURRENT_TIMESTAMP WHERE work_id=?""",
            [(path, path, work_id) for path, work_id in normalized],
        )
        self.connection.commit()
