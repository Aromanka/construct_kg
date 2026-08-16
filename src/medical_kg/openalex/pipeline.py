from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from medical_kg.openalex.catalog import OpenAlexCatalog
from medical_kg.openalex.filtering import WorkFilter
from medical_kg.openalex.fulltext import FullTextResolver
from medical_kg.openalex.models import OpenAlexWork, normalize_work_id
from medical_kg.openalex.screening import WorkScreener
from medical_kg.openalex.snapshot import OpenAlexSnapshot


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


@dataclass
class MaterializeStatistics:
    selected: int = 0
    written: int = 0
    fulltext: int = 0
    abstract_fallback: int = 0
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
                await screener.screen(
                    [work for work, _ in batch], options.llm_instruction or ""
                )
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
                if (
                    options.max_selected is not None
                    and statistics.selected >= options.max_selected
                ):
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
            return SelectionStatistics(
                selected=len(existing), explicit_selected=len(existing)
            )
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
        content_mode: str = "fulltext-or-abstract",
        limit: int | None = None,
    ) -> MaterializeStatistics:
        if content_mode not in {"fulltext", "abstract", "fulltext-or-abstract"}:
            raise ValueError("Invalid content_mode")
        output_dir = self.workspace / "documents"
        output_dir.mkdir(parents=True, exist_ok=True)
        statistics = MaterializeStatistics()
        for row in self.catalog.selected_rows(limit=limit):
            statistics.selected += 1
            work = OpenAlexWork.from_raw(
                json.loads(row["raw_json"]),
                snapshot_file=row["snapshot_file"],
                snapshot_line=row["snapshot_line"],
            )
            resolved = await resolver.resolve(work) if content_mode != "abstract" else None
            full_text = resolved.text if resolved else None
            if full_text:
                statistics.fulltext += 1
            if content_mode == "fulltext":
                content = full_text
            elif content_mode == "abstract":
                content = work.abstract
            else:
                content = full_text or work.abstract
                if not full_text and work.abstract:
                    statistics.abstract_fallback += 1
            if not content or not content.strip():
                statistics.skipped_no_content += 1
                self.catalog.set_materialized(
                    work.work_id,
                    fulltext_path=str(resolved.path) if resolved and resolved.path else None,
                    fulltext_status=resolved.status if resolved else "not_requested",
                    materialized_path=None,
                )
                continue
            document_path = output_dir / f"{work.work_id}.json"
            payload: dict[str, Any] = {
                "document_id": work.document_id,
                "openalex_work_id": work.work_id,
                "openalex_id": work.openalex_id,
                "title": work.title,
                "abstract": work.abstract,
                "full_text": full_text,
                "content": content,
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
            self.catalog.set_materialized(
                work.work_id,
                fulltext_path=str(resolved.path) if resolved and resolved.path else None,
                fulltext_status=resolved.status if resolved else "not_requested",
                materialized_path=str(document_path),
            )
            statistics.written += 1
        return statistics
