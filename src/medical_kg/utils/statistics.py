from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import func, select

from medical_kg.db.models import (
    Assertion,
    Document,
    DocumentRevision,
    Entity,
    EntityMention,
    ExtractionRun,
    InvalidRecord,
    ProcessingJob,
    RawAssertion,
    RelationType,
)
from medical_kg.db.repository import KnowledgeRepository


def assemble_knowledge_statistics(
    *,
    totals: Mapping[str, int],
    source_counts: Sequence[tuple[str, int]],
    job_counts: Sequence[tuple[str, str, str, int]],
    pass_counts: Sequence[tuple[str, int]],
    entity_type_counts: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    """Build the stable, JSON-friendly representation used by the CLI."""

    jobs_by_status: defaultdict[str, int] = defaultdict(int)
    job_details = []
    for stage, stage_version, status, count in job_counts:
        jobs_by_status[status] += int(count)
        job_details.append(
            {
                "stage": stage,
                "stage_version": stage_version,
                "status": status,
                "count": int(count),
            }
        )

    raw_assertions = int(totals["raw_assertions"])
    evidence_validated = int(totals["evidence_validated"])

    return {
        "documents": {
            "total": int(totals["documents"]),
            "revisions": int(totals["document_revisions"]),
            "by_source_type": {name: int(count) for name, count in source_counts},
        },
        "processing_jobs": {
            "total": int(totals["processing_jobs"]),
            "by_status": dict(sorted(jobs_by_status.items())),
            "details": job_details,
        },
        "bronze_knowledge": {
            "extraction_runs": int(totals["extraction_runs"]),
            "documents_with_completed_extraction": int(totals["extracted_documents"]),
            "extraction_runs_by_pass": {name: int(count) for name, count in pass_counts},
            "entity_mentions": int(totals["entity_mentions"]),
            "unique_mention_texts": int(totals["unique_mention_texts"]),
            "entity_mentions_by_type": {
                name: int(count) for name, count in entity_type_counts
            },
            "raw_assertions": raw_assertions,
            "documents_with_assertions": int(totals["assertion_documents"]),
            "evidence_validated": evidence_validated,
            "evidence_not_validated": raw_assertions - evidence_validated,
        },
        "canonical_knowledge": {
            "entities": int(totals["entities"]),
            "assertions": int(totals["assertions"]),
            "relation_types": int(totals["relation_types"]),
        },
        "quality": {
            "invalid_records": int(totals["invalid_records"]),
            "evidence_validation_rate": (
                round(evidence_validated / raw_assertions, 4) if raw_assertions else None
            ),
        },
    }


async def collect_knowledge_statistics(
    repository: KnowledgeRepository,
) -> dict[str, Any]:
    """Return committed pipeline counts without changing jobs or knowledge records."""

    def count(model: type[Any]):
        return select(func.count()).select_from(model).scalar_subquery()

    totals_statement = select(
        count(Document).label("documents"),
        count(DocumentRevision).label("document_revisions"),
        count(ProcessingJob).label("processing_jobs"),
        count(ExtractionRun).label("extraction_runs"),
        select(func.count(func.distinct(ExtractionRun.document_id)))
        .scalar_subquery()
        .label("extracted_documents"),
        count(EntityMention).label("entity_mentions"),
        select(func.count(func.distinct(func.lower(EntityMention.mention_text))))
        .scalar_subquery()
        .label("unique_mention_texts"),
        count(RawAssertion).label("raw_assertions"),
        select(func.count())
        .select_from(RawAssertion)
        .where(RawAssertion.evidence_validated.is_(True))
        .scalar_subquery()
        .label("evidence_validated"),
        select(func.count(func.distinct(RawAssertion.document_id)))
        .scalar_subquery()
        .label("assertion_documents"),
        count(Entity).label("entities"),
        count(Assertion).label("assertions"),
        count(RelationType).label("relation_types"),
        count(InvalidRecord).label("invalid_records"),
    )
    source_statement = (
        select(Document.source_type, func.count())
        .group_by(Document.source_type)
        .order_by(Document.source_type)
    )
    job_statement = (
        select(
            ProcessingJob.stage,
            ProcessingJob.stage_version,
            ProcessingJob.status,
            func.count(),
        )
        .group_by(ProcessingJob.stage, ProcessingJob.stage_version, ProcessingJob.status)
        .order_by(ProcessingJob.stage, ProcessingJob.stage_version, ProcessingJob.status)
    )
    pass_statement = (
        select(ExtractionRun.pass_name, func.count())
        .group_by(ExtractionRun.pass_name)
        .order_by(ExtractionRun.pass_name)
    )
    entity_type_statement = (
        select(EntityMention.entity_type, func.count())
        .group_by(EntityMention.entity_type)
        .order_by(EntityMention.entity_type)
    )

    async with repository.sessions() as session:
        totals = dict((await session.execute(totals_statement)).mappings().one())
        source_counts = list((await session.execute(source_statement)).all())
        job_counts = list((await session.execute(job_statement)).all())
        pass_counts = list((await session.execute(pass_statement)).all())
        entity_type_counts = list((await session.execute(entity_type_statement)).all())

    return assemble_knowledge_statistics(
        totals=totals,
        source_counts=source_counts,
        job_counts=job_counts,
        pass_counts=pass_counts,
        entity_type_counts=entity_type_counts,
    )
