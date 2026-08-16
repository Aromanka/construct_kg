from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from medical_kg.openalex.catalog import OpenAlexCatalog
from medical_kg.openalex.filtering import WorkFilter
from medical_kg.openalex.fulltext import FullTextResolver
from medical_kg.openalex.models import OpenAlexWork, normalize_work_id
from medical_kg.openalex.screening import WorkScreener
from medical_kg.openalex.snapshot import OpenAlexSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionOptions:
    work_filter: WorkFilter = field(default_factory=WorkFilter)
    llm_instruction: str | None = None
    llm_batch_size: int = 20
    include_ids: set[str] = field(default_factory=set)
    max_works: int | None = None
    max_candidates: int | None = None
    max_selected: int | None = None

    def __post_init__(self) -> None:
        if self.llm_batch_size < 1:
            raise ValueError("llm_batch_size must be at least 1")


@dataclass
class SelectionStatistics:
    scanned: int = 0
    candidates: int = 0
    selected: int = 0
    explicit_selected: int = 0
    selected_with_abstract: int = 0
    selected_without_abstract: int = 0
    selected_with_fulltext_hint: int = 0
    selected_without_fulltext_hint: int = 0
    selected_with_abstract_and_fulltext_hint: int = 0
    snapshot_parts_failed: int = 0
    snapshot_failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MaterializeStatistics:
    selected: int = 0
    written: int = 0
    fulltext: int = 0
    abstract_fallback: int = 0
    abstracts: int = 0
    abstract_batches: int = 0
    skipped_processed: int = 0
    download_failed: int = 0
    skipped_no_content: int = 0


class OpenAlexPipeline:
    def __init__(
        self,
        *,
        snapshot: OpenAlexSnapshot,
        catalog: OpenAlexCatalog,
        workspace: Path,
    ) -> None:
        self.snapshot = snapshot
        self.catalog = catalog
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    async def select(
        self, options: SelectionOptions, *, screener: WorkScreener | None = None
    ) -> SelectionStatistics:
        if options.llm_instruction and screener is None:
            raise ValueError("An LLM screener is required when llm_instruction is set")
        include_ids = {normalize_work_id(value) for value in options.include_ids}
        pending_ids = set(include_ids)
        statistics = SelectionStatistics()
        batch: list[tuple[OpenAlexWork, list[str]]] = []

        def record_selected_content(work: OpenAlexWork) -> None:
            has_abstract = bool(work.abstract and work.abstract.strip())
            if has_abstract:
                statistics.selected_with_abstract += 1
            else:
                statistics.selected_without_abstract += 1
            if work.has_fulltext_hint:
                statistics.selected_with_fulltext_hint += 1
            else:
                statistics.selected_without_fulltext_hint += 1
            if has_abstract and work.has_fulltext_hint:
                statistics.selected_with_abstract_and_fulltext_hint += 1

        run_id = self.catalog.start_run(
            {
                "filter": asdict(options.work_filter),
                "llm_instruction": options.llm_instruction,
                "llm_batch_size": options.llm_batch_size,
                "include_ids": sorted(include_ids),
                "max_works": options.max_works,
                "max_candidates": options.max_candidates,
                "max_selected": options.max_selected,
            }
        )

        async def flush() -> None:
            if not batch:
                return
            selected_indexes = (
                await screener.screen([work for work, _ in batch], options.llm_instruction or "")
                if options.llm_instruction and screener
                else set(range(len(batch)))
            )
            for index, (work, keywords) in enumerate(batch):
                selected = index in selected_indexes
                if (
                    selected
                    and options.max_selected is not None
                    and statistics.selected >= options.max_selected
                ):
                    selected = False
                self.catalog.upsert_work(
                    work,
                    selected=selected,
                    selection_method="llm" if options.llm_instruction else "filter",
                    matched_keywords=keywords,
                    commit=False,
                )
                if selected:
                    statistics.selected += 1
                    record_selected_content(work)
            self.catalog.connection.commit()
            batch.clear()

        try:
            for work in self.snapshot.iter_works(max_works=options.max_works):
                statistics.scanned += 1
                if work.work_id in pending_ids:
                    self.catalog.upsert_work(
                        work,
                        selected=True,
                        selection_method="explicit",
                        matched_keywords=[],
                    )
                    pending_ids.remove(work.work_id)
                    statistics.selected += 1
                    statistics.explicit_selected += 1
                    record_selected_content(work)
                    if not pending_ids and options.max_candidates == 0:
                        break
                    continue
                if (
                    options.max_candidates is not None
                    and statistics.candidates >= options.max_candidates
                ):
                    if not pending_ids:
                        break
                    continue
                if options.max_selected is not None and statistics.selected >= options.max_selected:
                    if not pending_ids:
                        break
                    continue
                matched, keywords = options.work_filter.match(work)
                if not matched:
                    continue
                statistics.candidates += 1
                batch.append((work, keywords))
                if len(batch) >= options.llm_batch_size:
                    await flush()
                if (
                    options.max_candidates is not None
                    and statistics.candidates >= options.max_candidates
                ):
                    await flush()
                    if not pending_ids:
                        break
                if options.max_selected is not None and statistics.selected >= options.max_selected:
                    if not pending_ids:
                        break
            await flush()
        finally:
            failures = self.snapshot.failures_for("works")
            statistics.snapshot_parts_failed = len(failures)
            statistics.snapshot_failures = [asdict(failure) for failure in failures]
            self.catalog.finish_run(
                run_id,
                scanned=statistics.scanned,
                candidates=statistics.candidates,
                selected=statistics.selected,
            )
        if pending_ids:
            missing = ", ".join(sorted(pending_ids))
            raise LookupError(
                f"Explicit Work IDs were not found in the scanned snapshot: {missing}"
            )
        return statistics

    async def add(self, work_ids: set[str], *, max_works: int | None = None) -> SelectionStatistics:
        normalized = {normalize_work_id(value) for value in work_ids}
        existing = {work_id for work_id in normalized if self.catalog.get(work_id)}
        self.mark_manual(existing)
        missing = normalized - existing
        if not missing:
            return SelectionStatistics(selected=len(existing), explicit_selected=len(existing))
        result = await self.select(
            SelectionOptions(
                include_ids=missing,
                max_works=max_works,
                max_candidates=0,
            )
        )
        self.mark_manual(missing)
        result.selected += len(existing)
        result.explicit_selected += len(existing)
        return result

    def mark_manual(self, work_ids: set[str]) -> int:
        """Keep cataloged Works selected independently of later screening runs."""

        changed = 0
        for value in work_ids:
            work_id = normalize_work_id(value)
            cursor = self.catalog.connection.execute(
                """UPDATE works SET selected=1, selection_method='manual',
                   updated_at=CURRENT_TIMESTAMP WHERE work_id=?""",
                (work_id,),
            )
            changed += cursor.rowcount
        self.catalog.connection.commit()
        return changed

    def enrich_sources(self) -> int:
        wanted = self.catalog.selected_source_ids()
        found = 0
        if not wanted:
            return found
        for source, _, _ in self.snapshot.iter_raw("sources"):
            source_id = source.get("id")
            if source_id and str(source_id) in wanted:
                self.catalog.upsert_sources([source], full=True)
                wanted.remove(str(source_id))
                found += 1
                if not wanted:
                    break
        self.catalog.connection.commit()
        return found

    async def materialize(
        self,
        *,
        resolver: FullTextResolver,
        content_mode: str = "fulltext",
        abstract_chunk_size: int = 12000,
        limit: int | None = None,
    ) -> MaterializeStatistics:
        if content_mode not in {"fulltext", "abstract", "fulltext-or-abstract"}:
            raise ValueError("Invalid content_mode")
        if abstract_chunk_size < 1:
            raise ValueError("abstract_chunk_size must be at least 1")
        output_dir = self.workspace / "documents"
        output_dir.mkdir(parents=True, exist_ok=True)
        statistics = MaterializeStatistics()
        abstract_works: list[OpenAlexWork] = []
        for row in self.catalog.selected_rows(limit=limit):
            statistics.selected += 1
            work = OpenAlexWork.from_raw(
                json.loads(row["raw_json"]),
                snapshot_file=row["snapshot_file"],
                snapshot_line=row["snapshot_line"],
            )
            if content_mode == "abstract":
                if row["abstract_processed"]:
                    statistics.skipped_processed += 1
                elif work.abstract and work.abstract.strip():
                    abstract_works.append(work)
                else:
                    statistics.skipped_no_content += 1
                continue

            if row["fulltext_processed"]:
                statistics.skipped_processed += 1
                continue
            try:
                resolved = await resolver.resolve(work)
            except Exception as error:  # one bad Work must not stop a large snapshot batch
                logger.warning("Skipping full text for %s: %s", work.work_id, error)
                resolved = None
            full_text = resolved.text if resolved else None
            fulltext_status = resolved.status if resolved else "download_failed"
            if not full_text or not full_text.strip():
                if fulltext_status in {
                    "download_failed",
                    "quota_unavailable",
                    "unauthorized",
                }:
                    statistics.download_failed += 1
                self.catalog.set_fulltext_materialized(
                    work.work_id,
                    fulltext_path=(str(resolved.path) if resolved and resolved.path else None),
                    fulltext_status=fulltext_status,
                    materialized_path=None,
                    processed=False,
                )
                if (
                    content_mode == "fulltext-or-abstract"
                    and not row["abstract_processed"]
                    and work.abstract
                    and work.abstract.strip()
                ):
                    abstract_works.append(work)
                    statistics.abstract_fallback += 1
                else:
                    statistics.skipped_no_content += 1
                continue

            statistics.fulltext += 1
            document_path = output_dir / f"{work.work_id}.json"
            payload: dict[str, Any] = {
                "document_id": work.document_id,
                "openalex_work_id": work.work_id,
                "openalex_id": work.openalex_id,
                "title": work.title,
                "abstract": work.abstract,
                "full_text": full_text,
                "content": full_text,
                "content_mode": "fulltext",
                "doi": work.doi,
                "publication_year": work.publication_year,
                "language": work.language,
                "work_type": work.work_type,
                "sources": work.sources,
                "primary_source_id": work.primary_source_id,
                "primary_source_name": work.primary_source_name,
                "referenced_works": work.raw.get("referenced_works") or [],
                "snapshot_locator": {
                    "file": work.snapshot_file,
                    "line": work.snapshot_line,
                },
                "source_type": "research",
            }
            document_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.catalog.set_fulltext_materialized(
                work.work_id,
                fulltext_path=str(resolved.path) if resolved.path else None,
                fulltext_status=resolved.status,
                materialized_path=str(document_path),
                processed=True,
            )
            statistics.written += 1

        if abstract_works:
            self._materialize_abstract_batches(
                abstract_works,
                output_dir=output_dir,
                chunk_size=abstract_chunk_size,
                statistics=statistics,
            )
        return statistics

    @staticmethod
    def _abstract_section(work: OpenAlexWork) -> str:
        return (
            f"[OPENALEX WORK {work.work_id}]\n"
            f"Title: {work.title}\n"
            f"DOI: {work.doi or ''}\n"
            f"Abstract:\n{(work.abstract or '').strip()}\n"
            f"[/OPENALEX WORK {work.work_id}]"
        )

    def _materialize_abstract_batches(
        self,
        works: list[OpenAlexWork],
        *,
        output_dir: Path,
        chunk_size: int,
        statistics: MaterializeStatistics,
    ) -> None:
        batch_dir = output_dir / "abstract_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch: list[tuple[OpenAlexWork, str]] = []
        batch_length = 0

        def flush() -> None:
            nonlocal batch_length
            if not batch:
                return
            content = "\n\n".join(section for _, section in batch)
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
            document_path = batch_dir / f"abstract_batch_{digest}.json"
            batch_works = [work for work, _ in batch]
            payload: dict[str, Any] = {
                "document_id": f"openalex:abstract-batch:{digest}",
                "title": f"OpenAlex abstract batch ({len(batch_works)} works)",
                "content": content,
                "content_mode": "abstract",
                "openalex_work_ids": [work.work_id for work in batch_works],
                "works": [
                    {
                        "document_id": work.document_id,
                        "openalex_work_id": work.work_id,
                        "openalex_id": work.openalex_id,
                        "title": work.title,
                        "abstract": work.abstract,
                        "doi": work.doi,
                        "publication_year": work.publication_year,
                        "sources": work.sources,
                        "primary_source_id": work.primary_source_id,
                        "primary_source_name": work.primary_source_name,
                        "snapshot_locator": {
                            "file": work.snapshot_file,
                            "line": work.snapshot_line,
                        },
                    }
                    for work in batch_works
                ],
                "source_type": "research",
            }
            document_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.catalog.set_abstract_materialized(
                (work.work_id for work in batch_works),
                materialized_path=str(document_path),
            )
            statistics.abstracts += len(batch_works)
            statistics.abstract_batches += 1
            statistics.written += 1
            batch.clear()
            batch_length = 0

        for work in works:
            section = self._abstract_section(work)
            separator_length = 2 if batch else 0
            if batch and batch_length + separator_length + len(section) > chunk_size:
                flush()
                separator_length = 0
            batch.append((work, section))
            batch_length += separator_length + len(section)
            # An unusually long single abstract is kept intact so article boundaries
            # and provenance are not lost; downstream character chunking can split it.
            if len(section) >= chunk_size:
                flush()
        flush()
