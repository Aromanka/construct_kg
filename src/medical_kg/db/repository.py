from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from medical_kg.db.models import (
    Base,
    Document,
    DocumentRevision,
    EntityMention,
    ExtractionRun,
    ProcessingJob,
    RawAssertion,
    RelationType,
)
from medical_kg.models.assertion import ExtractionOutput
from medical_kg.models.document import DocumentInput
from medical_kg.models.job import JobStatus


@dataclass(frozen=True)
class ClaimedJob:
    job_id: uuid.UUID
    document_id: str
    stage: str
    stage_version: str
    retry_count: int
    content: str
    content_hash: str


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
    """Transactional access to the authoritative PostgreSQL knowledge store."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def register_document(self, document: DocumentInput) -> tuple[bool, bool]:
        """Return ``(created, changed)`` and invalidate jobs if the content changed."""
        async with self.sessions.begin() as session:
            current = await session.get(Document, document.document_id, with_for_update=True)
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
                    )
                )
                session.add(
                    DocumentRevision(
                        document_id=document.document_id,
                        content_hash=document.content_hash,
                        content=document.content,
                        file_path=str(document.file_path),
                    )
                )
                return True, True

            changed = current.content_hash != document.content_hash
            current.file_path = str(document.file_path)
            current.title = document.title or current.title
            current.doi = document.doi or current.doi
            current.pmid = document.pmid or current.pmid
            if changed:
                current.content = document.content
                current.content_hash = document.content_hash
                session.add(
                    DocumentRevision(
                        document_id=document.document_id,
                        content_hash=document.content_hash,
                        content=document.content,
                        file_path=str(document.file_path),
                    )
                )
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
        async with self.sessions.begin() as session:
            result = await session.execute(statement)
            return result.rowcount or 0

    async def claim_job(
        self,
        *,
        stages: Sequence[str],
        stage_version: str,
        worker_id: str,
        document_id: str | None = None,
    ) -> ClaimedJob | None:
        """Atomically claim one pending job using PostgreSQL SKIP LOCKED."""
        async with self.sessions.begin() as session:
            statement = (
                select(ProcessingJob)
                .where(
                    ProcessingJob.status == JobStatus.PENDING.value,
                    ProcessingJob.stage.in_(stages),
                    ProcessingJob.stage_version == stage_version,
                )
                .order_by(ProcessingJob.job_id)
                .with_for_update(skip_locked=True)
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
            job.started_at = datetime.now(UTC)
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
        async with self.sessions.begin() as session:
            locked_job = await session.get(ProcessingJob, job.job_id, with_for_update=True)
            if locked_job is None or locked_job.status != JobStatus.RUNNING.value:
                raise RuntimeError(f"Job {job.job_id} is not RUNNING")
            run = ExtractionRun(
                **run_spec.__dict__,
                document_id=job.document_id,
                content_hash=job.content_hash,
            )
            session.add(run)
            await session.flush()

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
                )
                object_ = EntityMention(
                    document_id=job.document_id,
                    extraction_run_id=run.extraction_run_id,
                    mention_text=assertion.object.mention,
                    entity_type=assertion.object.entity_type.value,
                    entity_type_detail=assertion.object.entity_type_detail,
                    character_start=object_start,
                    character_end=object_end,
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
                        raw_llm_output=raw_output,
                        evidence_validated=evidence_valid,
                        validation_error=(
                            None if evidence_valid else "Evidence is not exact source text"
                        ),
                    )
                )

            locked_job.status = JobStatus.SUCCESS.value
            locked_job.finished_at = datetime.now(UTC)
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
        async with self.sessions.begin() as session:
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                return
            job.status = JobStatus.FAILED.value
            job.retry_count += 1
            job.finished_at = datetime.now(UTC)
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
                finished_at=None,
                error_message=None,
            )
        )
        if stage_prefix:
            statement = statement.where(ProcessingJob.stage.startswith(stage_prefix))
        if document_id:
            statement = statement.where(ProcessingJob.document_id == document_id)
        async with self.sessions.begin() as session:
            result = await session.execute(statement)
            return result.rowcount or 0

    async def recover_stale_jobs(self, *, older_than_seconds: float) -> int:
        """Return abandoned RUNNING jobs to PENDING without disturbing active workers."""
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        statement = (
            update(ProcessingJob)
            .where(
                ProcessingJob.status == JobStatus.RUNNING.value,
                ProcessingJob.started_at < cutoff,
            )
            .values(
                status=JobStatus.PENDING.value,
                worker_id=None,
                started_at=None,
                finished_at=None,
                error_message="Recovered abandoned RUNNING job",
            )
        )
        async with self.sessions.begin() as session:
            result = await session.execute(statement)
            return result.rowcount or 0

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
        async with self.sessions.begin() as session:
            result = await session.execute(statement)
            return result.rowcount or 0
