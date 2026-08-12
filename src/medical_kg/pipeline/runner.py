from __future__ import annotations

from pathlib import Path

from medical_kg.bronze.extraction import BronzeExtractor, RunStatistics
from medical_kg.db.repository import KnowledgeRepository
from medical_kg.landing.loader import DocumentLoader, IngestResult
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

    async def ingest(self, source: Path) -> IngestResult:
        return await self.loader.ingest(source)

    async def extract(
        self,
        *,
        limit: int | None = None,
        document_id: str | None = None,
        worker_id: str | None = None,
    ) -> RunStatistics:
        # Twice the worst configured provider window avoids stealing legitimate long retries while
        # ensuring a worker crash does not leave work permanently stuck in RUNNING.
        stale_after = 2 * (
            self.extractor.settings.processing.request_timeout
            * (self.extractor.settings.processing.max_retries + 1)
            + 60 * self.extractor.settings.processing.max_retries
        )
        await self.repository.recover_stale_jobs(older_than_seconds=stale_after)
        document_ids = (
            [document_id]
            if document_id
            else await self.repository.list_document_ids(limit)
        )
        await self.extractor.enqueue(document_ids)
        return await self.extractor.run(
            worker_id=worker_id or default_worker_id(), document_id=document_id
        )

    async def run(
        self,
        source: Path | None = None,
        *,
        limit: int | None = None,
        document_id: str | None = None,
        worker_id: str | None = None,
    ) -> tuple[IngestResult | None, RunStatistics]:
        ingest_result = await self.ingest(source) if source else None
        extract_result = await self.extract(
            limit=limit, document_id=document_id, worker_id=worker_id
        )
        return ingest_result, extract_result
