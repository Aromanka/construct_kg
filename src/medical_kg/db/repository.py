from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from medical_kg.db.models import (
    APIRateLimit,
    Base,
    Document,
    DocumentRevision,
    EntityMention,
    ExtractionChunk,
    ExtractionRun,
    ProcessingJob,
    RawAssertion,
    RelationType,
)
from medical_kg.models.assertion import ExtractionOutput
from medical_kg.models.document import DocumentInput
from medical_kg.models.job import JobStatus
from medical_kg.models.source import KnowledgeSource


@dataclass(frozen=True)
class ClaimedJob:
    job_id: uuid.UUID
    document_id: str
    stage: str
    stage_version: str
    retry_count: int
    content: str
    content_hash: str
    source_type: str
    worker_id: str = ""


@dataclass(frozen=True)
class StoredChunkResult:
    chunk_index: int
    validated_output: dict[str, Any]
    raw_output: dict[str, Any]
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ExtractionRunSpec:
    model_provider: str
    model_name: str
    prompt_name: str
    prompt_version: str
    pass_name: str
    temperature: float
    code_version: str


class KnowledgeRepository:
    """Transactional access to the authoritative SQLite knowledge store."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._write_lock = asyncio.Lock()

    @asynccontextmanager
    async def _write_session(self) -> AsyncIterator[AsyncSession]:
        """Serialize SQLite writers and make read-modify-write operations atomic."""
        async with self._write_lock:
            async with self.sessions() as session:
                await session.execute(text("BEGIN IMMEDIATE"))
                try:
                    yield session
                except BaseException:
                    await session.rollback()
                    raise
                else:
                    await session.commit()

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def register_document(self, document: DocumentInput) -> tuple[bool, bool]:
        """Return ``(created, changed)`` and invalidate jobs if the content changed."""
        async with self._write_session() as session:
            current = await session.get(Document, document.document_id)
            if current is None:
                session.add(
                    Document(
                        document_id=document.document_id,
                        file_path=str(document.file_path),
                        title=document.title,
                        doi=document.doi,
                        pmid=document.pmid,
                        content=document.content,
                        content_hash=document.content_hash,
                        source_type=document.source_type.value,
                    )
                )
                session.add(
                    DocumentRevision(
                        document_id=document.document_id,
                        content_hash=document.content_hash,
                        content=document.content,
                        file_path=str(document.file_path),
                        source_type=document.source_type.value,
                    )
                )
                return True, True

            content_changed = current.content_hash != document.content_hash
            source_type_changed = current.source_type != document.source_type.value
            changed = content_changed or source_type_changed
            current.file_path = str(document.file_path)
            current.title = document.title or current.title
            current.doi = document.doi or current.doi
            current.pmid = document.pmid or current.pmid
            current.source_type = document.source_type.value
            if content_changed:
                current.content = document.content
                current.content_hash = document.content_hash
                session.add(
                    DocumentRevision(
                        document_id=document.document_id,
                        content_hash=document.content_hash,
                        content=document.content,
                        file_path=str(document.file_path),
                        source_type=document.source_type.value,
                    )
                )
            if changed:
                # Bronze history remains intact. Jobs are removed so current content gets new runs.
                await session.execute(
                    delete(ProcessingJob).where(ProcessingJob.document_id == document.document_id)
                )
            return False, changed

    async def list_document_ids(self, limit: int | None = None) -> list[str]:
        statement = select(Document.document_id).order_by(Document.document_id)
        if limit is not None:
            statement = statement.limit(limit)
        async with self.sessions() as session:
            return list((await session.scalars(statement)).all())

    async def enqueue_jobs(
        self,
        *,
        document_ids: Sequence[str],
        stages: Sequence[str],
        stage_version: str,
    ) -> int:
        rows = [
            {
                "document_id": document_id,
                "stage": stage,
                "stage_version": stage_version,
                "status": JobStatus.PENDING.value,
            }
            for document_id in document_ids
            for stage in stages
        ]
        if not rows:
            return 0
        statement = insert(ProcessingJob).values(rows)
        statement = statement.on_conflict_do_nothing(
            index_elements=["document_id", "stage", "stage_version"]
        )
        async with self._write_session() as session:
            result = await session.execute(statement)
            return result.rowcount or 0

    async def count_pending_jobs(
        self,
        *,
        stages: Sequence[str],
        stage_version: str,
        document_id: str | None = None,
    ) -> int:
        statement = (
            select(func.count())
            .select_from(ProcessingJob)
            .where(
                ProcessingJob.status == JobStatus.PENDING.value,
                ProcessingJob.stage.in_(stages),
                ProcessingJob.stage_version == stage_version,
            )
        )
        if document_id:
            statement = statement.where(ProcessingJob.document_id == document_id)
        async with self.sessions() as session:
            return int((await session.scalar(statement)) or 0)

    async def claim_job(
        self,
        *,
        stages: Sequence[str],
        stage_version: str,
        worker_id: str,
        document_id: str | None = None,
        lease_seconds: float = 900.0,
    ) -> ClaimedJob | None:
        """Atomically claim one pending job under SQLite's immediate write lock."""
        async with self._write_session() as session:
            statement = (
                select(ProcessingJob)
                .where(
                    ProcessingJob.status == JobStatus.PENDING.value,
                    ProcessingJob.stage.in_(stages),
                    ProcessingJob.stage_version == stage_version,
                )
                .order_by(ProcessingJob.job_id)
                .limit(1)
            )
            if document_id:
                statement = statement.where(ProcessingJob.document_id == document_id)
            job = await session.scalar(statement)
            if job is None:
                return None
            document = await session.get(Document, job.document_id)
            if document is None:
                raise RuntimeError(f"Job {job.job_id} refers to missing document {job.document_id}")
            job.status = JobStatus.RUNNING.value
            job.worker_id = worker_id
            now = datetime.now(timezone.utc)
            job.started_at = now
            job.heartbeat_at = now
            job.lease_expires_at = now + timedelta(seconds=lease_seconds)
            job.finished_at = None
            job.error_message = None
            return ClaimedJob(
                job_id=job.job_id,
                document_id=job.document_id,
                stage=job.stage,
                stage_version=job.stage_version,
                retry_count=job.retry_count,
                content=document.content,
                content_hash=document.content_hash,
                source_type=document.source_type,
                worker_id=worker_id,
            )

    async def heartbeat_job(
        self, job_id: uuid.UUID, *, worker_id: str, lease_seconds: float
    ) -> bool:
        now = datetime.now(timezone.utc)
        statement = (
            update(ProcessingJob)
            .where(
                ProcessingJob.job_id == job_id,
                ProcessingJob.status == JobStatus.RUNNING.value,
                ProcessingJob.worker_id == worker_id,
            )
            .values(
                heartbeat_at=now,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
        )
        async with self._write_session() as session:
            result = await session.execute(statement)
            return bool(result.rowcount)

    async def prepare_chunks(
        self,
        *,
        job: ClaimedJob,
        pass_name: str,
        chunks: Sequence[Any],
    ) -> dict[int, StoredChunkResult]:
        rows = [
            {
                "job_id": job.job_id,
                "document_id": job.document_id,
                "pass_name": pass_name,
                "stage_version": job.stage_version,
                "content_hash": job.content_hash,
                "chunk_index": chunk.index,
                "character_start": chunk.character_start,
                "character_end": chunk.character_end,
                "status": JobStatus.PENDING.value,
            }
            for chunk in chunks
        ]
        async with self._write_session() as session:
            if rows:
                statement = insert(ExtractionChunk).values(rows).on_conflict_do_nothing(
                    index_elements=[
                        "document_id",
                        "pass_name",
                        "stage_version",
                        "content_hash",
                        "chunk_index",
                    ]
                )
                await session.execute(statement)
            await session.execute(
                update(ExtractionChunk)
                .where(
                    ExtractionChunk.job_id == job.job_id,
                    ExtractionChunk.status != JobStatus.SUCCESS.value,
                )
                .values(
                    status=JobStatus.PENDING.value,
                    worker_id=None,
                    started_at=None,
                    finished_at=None,
                    error_message=None,
                )
            )
            successful = (
                await session.scalars(
                    select(ExtractionChunk).where(
                        ExtractionChunk.job_id == job.job_id,
                        ExtractionChunk.status == JobStatus.SUCCESS.value,
                    )
                )
            ).all()
        return {
            chunk.chunk_index: StoredChunkResult(
                chunk_index=chunk.chunk_index,
                validated_output=chunk.validated_output or {},
                raw_output=chunk.raw_output or {},
                input_tokens=chunk.input_tokens,
                output_tokens=chunk.output_tokens,
            )
            for chunk in successful
        }

    async def start_chunk(self, job_id: uuid.UUID, chunk_index: int, worker_id: str) -> None:
        async with self._write_session() as session:
            await session.execute(
                update(ExtractionChunk)
                .where(
                    ExtractionChunk.job_id == job_id,
                    ExtractionChunk.chunk_index == chunk_index,
                    ExtractionChunk.status != JobStatus.SUCCESS.value,
                )
                .values(
                    status=JobStatus.RUNNING.value,
                    worker_id=worker_id,
                    started_at=datetime.now(timezone.utc),
                    finished_at=None,
                    error_message=None,
                )
            )

    async def complete_chunk(
        self, job_id: uuid.UUID, chunk_index: int, response: Any
    ) -> None:
        statement = (
            update(ExtractionChunk)
            .where(
                ExtractionChunk.job_id == job_id,
                ExtractionChunk.chunk_index == chunk_index,
                ExtractionChunk.status == JobStatus.RUNNING.value,
            )
            .values(
                status=JobStatus.SUCCESS.value,
                finished_at=datetime.now(timezone.utc),
                error_message=None,
                validated_output=response.output.model_dump(mode="json"),
                raw_output=response.raw_output,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        )
        async with self._write_session() as session:
            result = await session.execute(statement)
            if not result.rowcount:
                raise RuntimeError(
                    f"Chunk {chunk_index} for job {job_id} is not RUNNING"
                )

    async def fail_chunk(self, job_id: uuid.UUID, chunk_index: int, error: str) -> None:
        async with self._write_session() as session:
            await session.execute(
                update(ExtractionChunk)
                .where(
                    ExtractionChunk.job_id == job_id,
                    ExtractionChunk.chunk_index == chunk_index,
                    ExtractionChunk.status != JobStatus.SUCCESS.value,
                )
                .values(
                    status=JobStatus.FAILED.value,
                    retry_count=ExtractionChunk.retry_count + 1,
                    finished_at=datetime.now(timezone.utc),
                    error_message=error[:20_000],
                )
            )

    async def complete_extraction(
        self,
        *,
        job: ClaimedJob,
        run_spec: ExtractionRunSpec,
        output: ExtractionOutput,
        raw_output: dict[str, Any],
    ) -> uuid.UUID:
        """Persist one Bronze pass and mark its job successful in one transaction."""
        async with self._write_session() as session:
            locked_job = await session.get(ProcessingJob, job.job_id)
            if locked_job is None or locked_job.status != JobStatus.RUNNING.value:
                raise RuntimeError(f"Job {job.job_id} is not RUNNING")
            run = ExtractionRun(
                **run_spec.__dict__,
                document_id=job.document_id,
                content_hash=job.content_hash,
                source_type=job.source_type,
                raw_llm_output=raw_output,
            )
            session.add(run)
            await session.flush()

            source = KnowledgeSource(
                document_id=job.document_id, source_type=job.source_type
            ).model_dump(mode="json")
            sources = [source]
            for assertion in output.assertions:
                evidence_start = job.content.find(assertion.evidence_text)
                evidence_valid = evidence_start >= 0
                subject_start, subject_end = self._locate_mention(
                    job.content, assertion.evidence_text, assertion.subject.mention, evidence_start
                )
                object_start, object_end = self._locate_mention(
                    job.content, assertion.evidence_text, assertion.object.mention, evidence_start
                )
                subject = EntityMention(
                    document_id=job.document_id,
                    extraction_run_id=run.extraction_run_id,
                    mention_text=assertion.subject.mention,
                    entity_type=assertion.subject.entity_type.value,
                    entity_type_detail=assertion.subject.entity_type_detail,
                    character_start=subject_start,
                    character_end=subject_end,
                    sources=sources,
                )
                object_ = EntityMention(
                    document_id=job.document_id,
                    extraction_run_id=run.extraction_run_id,
                    mention_text=assertion.object.mention,
                    entity_type=assertion.object.entity_type.value,
                    entity_type_detail=assertion.object.entity_type_detail,
                    character_start=object_start,
                    character_end=object_end,
                    sources=sources,
                )
                session.add_all([subject, object_])
                await session.flush()
                session.add(
                    RawAssertion(
                        document_id=job.document_id,
                        extraction_run_id=run.extraction_run_id,
                        subject_mention_id=subject.mention_id,
                        object_mention_id=object_.mention_id,
                        subject_mention=assertion.subject.mention,
                        subject_type=assertion.subject.entity_type.value,
                        object_mention=assertion.object.mention,
                        object_type=assertion.object.entity_type.value,
                        detailed_relation=assertion.detailed_relation,
                        llm_confidence=assertion.llm_confidence,
                        evidence_text=assertion.evidence_text,
                        qualifiers=assertion.qualifiers.model_dump(mode="json", exclude_none=True),
                        negated=assertion.negated,
                        speculative=assertion.speculative,
                        evidence_validated=evidence_valid,
                        validation_error=(
                            None if evidence_valid else "Evidence is not exact source text"
                        ),
                        sources=sources,
                    )
                )

            locked_job.status = JobStatus.SUCCESS.value
            locked_job.finished_at = datetime.now(timezone.utc)
            locked_job.heartbeat_at = None
            locked_job.lease_expires_at = None
            locked_job.error_message = None
            return run.extraction_run_id

    @staticmethod
    def _locate_mention(
        content: str, evidence: str, mention: str, evidence_start: int
    ) -> tuple[int | None, int | None]:
        if evidence_start >= 0:
            relative = evidence.find(mention)
            if relative >= 0:
                start = evidence_start + relative
                return start, start + len(mention)
        start = content.find(mention)
        return (start, start + len(mention)) if start >= 0 else (None, None)

    async def fail_job(self, job_id: uuid.UUID, error: str) -> None:
        async with self._write_session() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED.value
            job.retry_count += 1
            job.finished_at = datetime.now(timezone.utc)
            job.heartbeat_at = None
            job.lease_expires_at = None
            job.error_message = error[:20_000]

    async def retry_failed(
        self, *, stage_prefix: str | None = None, document_id: str | None = None
    ) -> int:
        statement = (
            update(ProcessingJob)
            .where(ProcessingJob.status == JobStatus.FAILED.value)
            .values(
                status=JobStatus.PENDING.value,
                worker_id=None,
                started_at=None,
                heartbeat_at=None,
                lease_expires_at=None,
                finished_at=None,
                error_message=None,
            )
        )
        if stage_prefix:
            statement = statement.where(ProcessingJob.stage.startswith(stage_prefix))
        if document_id:
            statement = statement.where(ProcessingJob.document_id == document_id)
        async with self._write_session() as session:
            result = await session.execute(statement)
            return result.rowcount or 0

    async def recover_stale_jobs(self, *, older_than_seconds: float) -> int:
        """Return abandoned RUNNING jobs to PENDING without disturbing active workers."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=older_than_seconds)
        statement = (
            update(ProcessingJob)
            .where(
                ProcessingJob.status == JobStatus.RUNNING.value,
                or_(
                    ProcessingJob.lease_expires_at < now,
                    and_(
                        ProcessingJob.lease_expires_at.is_(None),
                        ProcessingJob.started_at < cutoff,
                    ),
                ),
            )
            .values(
                status=JobStatus.PENDING.value,
                worker_id=None,
                started_at=None,
                heartbeat_at=None,
                lease_expires_at=None,
                finished_at=None,
                error_message="Recovered abandoned RUNNING job",
            )
        )
        async with self._write_session() as session:
            result = await session.execute(statement)
            return result.rowcount or 0

    async def reserve_api_capacity(
        self,
        *,
        limiter_key: str,
        estimated_tokens: int,
        requests_per_minute: int,
        tokens_per_minute: int,
    ) -> float:
        """Reserve a smooth global request/token slot and return seconds to wait."""
        async with self._write_session() as session:
            now = datetime.now(timezone.utc)
            await session.execute(
                insert(APIRateLimit)
                .values(limiter_key=limiter_key, next_request_at=now, next_token_at=now)
                .on_conflict_do_nothing(index_elements=["limiter_key"])
            )
            limiter = await session.get(APIRateLimit, limiter_key)
            if limiter is None:
                raise RuntimeError(f"Unable to create API rate limiter {limiter_key!r}")
            scheduled_at = max(now, limiter.next_request_at, limiter.next_token_at)
            limiter.next_request_at = scheduled_at + timedelta(
                seconds=60 / requests_per_minute
            )
            limiter.next_token_at = scheduled_at + timedelta(
                seconds=60 * estimated_tokens / tokens_per_minute
            )
            return max(0.0, (scheduled_at - now).total_seconds())

    async def reconcile_api_tokens(
        self,
        *,
        limiter_key: str,
        estimated_tokens: int,
        actual_tokens: int,
        tokens_per_minute: int,
    ) -> None:
        difference = actual_tokens - estimated_tokens
        if difference == 0:
            return
        async with self._write_session() as session:
            now = datetime.now(timezone.utc)
            limiter = await session.get(APIRateLimit, limiter_key)
            if limiter is None:
                return
            adjusted = limiter.next_token_at + timedelta(
                seconds=60 * difference / tokens_per_minute
            )
            limiter.next_token_at = max(now, adjusted)

    async def job_status(self) -> list[dict[str, Any]]:
        statement = (
            select(
                ProcessingJob.stage,
                ProcessingJob.stage_version,
                ProcessingJob.status,
                func.count(),
            )
            .group_by(ProcessingJob.stage, ProcessingJob.stage_version, ProcessingJob.status)
            .order_by(ProcessingJob.stage, ProcessingJob.stage_version, ProcessingJob.status)
        )
        async with self.sessions() as session:
            rows = (await session.execute(statement)).all()
        return [
            {"stage": row[0], "stage_version": row[1], "status": row[2], "count": row[3]}
            for row in rows
        ]

    async def seed_relations(self, names: Sequence[str]) -> int:
        if not names:
            return 0
        statement = insert(RelationType).values([{"canonical_name": name} for name in names])
        statement = statement.on_conflict_do_nothing(index_elements=["canonical_name"])
        async with self._write_session() as session:
            result = await session.execute(statement)
            return result.rowcount or 0
