from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from medical_kg.bronze.extraction import BronzeExtractor, RunStatistics
from medical_kg.db.repository import KnowledgeRepository
from medical_kg.landing.loader import DocumentLoader, IngestResult
from medical_kg.models.source import SourceType
from medical_kg.pipeline.worker import default_worker_id


class PipelineRunner:
    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        loader: DocumentLoader,
        extractor: BronzeExtractor,
    ) -> None:
        self.repository = repository
        self.loader = loader
        self.extractor = extractor

    async def ingest(
        self, source: Path, *, source_type: SourceType = SourceType.RESEARCH
    ) -> IngestResult:
        return await self.loader.ingest(source, source_type=source_type)

    async def extract(
        self,
        *,
        limit: int | None = None,
        document_id: str | None = None,
        document_ids: Sequence[str] | None = None,
        worker_id: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int = 0,
        progress: Callable[[RunStatistics], None] | None = None,
        progress_total: Callable[[int], None] | None = None,
    ) -> RunStatistics:
        # Fail invalid chunk settings before touching job state in SQLite.
        self.extractor.stage_version(chunk_size, chunk_overlap)
        # Twice the worst configured provider window avoids stealing legitimate long retries while
        # ensuring a worker crash does not leave work permanently stuck in RUNNING.
        stale_after = 2 * (
            self.extractor.settings.processing.request_timeout
            * (self.extractor.settings.processing.max_retries + 1)
            + 60 * self.extractor.settings.processing.max_retries
        )
        await self.repository.recover_stale_jobs(older_than_seconds=stale_after)
        if document_id and document_ids is not None:
            raise ValueError("document_id and document_ids cannot be used together")
        selected_document_ids = (
            list(document_ids)
            if document_ids is not None
            else ([document_id] if document_id else await self.repository.list_document_ids(limit))
        )
        await self.extractor.enqueue(
            selected_document_ids, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        stage_version = self.extractor.stage_version(chunk_size, chunk_overlap)
        if progress_total is not None:
            progress_total(
                await self.repository.count_pending_jobs(
                    stages=self.extractor.job_stages,
                    stage_version=stage_version,
                    document_id=document_id,
                )
            )
        return await self.extractor.run(
            worker_id=worker_id or default_worker_id(),
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress=progress,
        )

    async def run(
        self,
        source: Path | None = None,
        *,
        limit: int | None = None,
        document_id: str | None = None,
        worker_id: str | None = None,
        source_type: SourceType = SourceType.RESEARCH,
        chunk_size: int | None = None,
        chunk_overlap: int = 0,
        ingest_progress: Callable[[IngestResult], None] | None = None,
        extraction_progress: Callable[[RunStatistics], None] | None = None,
        extraction_total: Callable[[int], None] | None = None,
    ) -> tuple[IngestResult | None, RunStatistics]:
        ingest_result = (
            await self.loader.ingest(
                source,
                source_type=source_type,
                progress=ingest_progress,
            )
            if source
            else None
        )
        extract_result = await self.extract(
            limit=limit,
            document_id=document_id,
            worker_id=worker_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress=extraction_progress,
            progress_total=extraction_total,
        )
        return ingest_result, extract_result
