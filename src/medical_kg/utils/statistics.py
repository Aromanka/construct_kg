from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import func, select

from medical_kg.db.models import (
    Assertion,
    AssertionEvidence,
    Document,
    DocumentRevision,
    Entity,
    EntityMention,
    EntityResolution,
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
    graph_quality: Mapping[str, Any] | None = None,
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
            "resolved_mentions": int(totals.get("resolved_mentions", 0)),
            "evidence_links": int(totals.get("evidence_links", 0)),
        },
        "quality": {
            "invalid_records": int(totals["invalid_records"]),
            "evidence_validation_rate": (
                round(evidence_validated / raw_assertions, 4) if raw_assertions else None
            ),
        },
        "graph_quality": dict(graph_quality or {}),
    }


def compute_graph_quality(
    *,
    edges: Sequence[tuple[str, str, str]],
    mention_count: int,
    entity_count: int,
    entity_documents: Sequence[tuple[str, str]],
    evidence_count: int,
) -> dict[str, Any]:
    """Compute connectivity metrics without rewarding unsafe entity merges."""

    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    degree: defaultdict[str, int] = defaultdict(int)
    other_count = 0
    for subject, object_, relation in edges:
        adjacency[subject].add(object_)
        adjacency[object_].add(subject)
        degree[subject] += 1
        degree[object_] += 1
        other_count += relation.casefold() == "other"

    nodes = set(adjacency)
    largest = 0
    unseen = set(nodes)
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            neighbours = adjacency[current] & unseen
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
        largest = max(largest, size)

    singleton_edges = sum(
        degree[subject] == 1 and degree[object_] == 1 for subject, object_, _ in edges
    )
    documents_by_entity: defaultdict[str, set[str]] = defaultdict(set)
    for entity_id, document_id in entity_documents:
        documents_by_entity[entity_id].add(document_id)
    reused = sum(len(documents) >= 2 for documents in documents_by_entity.values())
    edge_count = len(edges)
    return {
        "singleton_edge_ratio": round(singleton_edges / edge_count, 4)
        if edge_count
        else None,
        "largest_connected_component_ratio": round(largest / entity_count, 4)
        if entity_count
        else None,
        "canonical_compression_ratio": round(mention_count / entity_count, 4)
        if entity_count
        else None,
        "cross_document_reuse_rate": round(reused / entity_count, 4)
        if entity_count
        else None,
        "relation_other_ratio": round(other_count / edge_count, 4)
        if edge_count
        else None,
        "duplicate_canonical_assertion_ratio": round(
            max(0, evidence_count - edge_count) / evidence_count, 4
        )
        if evidence_count
        else None,
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
        count(EntityResolution).label("resolved_mentions"),
        count(AssertionEvidence).label("evidence_links"),
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
    edge_statement = (
        select(
            Assertion.subject_entity_id,
            Assertion.object_entity_id,
            RelationType.canonical_name,
        )
        .join(
            RelationType,
            RelationType.relation_id == Assertion.canonical_relation_id,
        )
        .order_by(Assertion.assertion_id)
    )
    entity_document_statement = (
        select(EntityResolution.entity_id, EntityMention.document_id)
        .join(
            EntityMention,
            EntityMention.mention_id == EntityResolution.mention_id,
        )
        .distinct()
    )

    async with repository.sessions() as session:
        totals = dict((await session.execute(totals_statement)).mappings().one())
        source_counts = list((await session.execute(source_statement)).all())
        job_counts = list((await session.execute(job_statement)).all())
        pass_counts = list((await session.execute(pass_statement)).all())
        entity_type_counts = list((await session.execute(entity_type_statement)).all())
        edges = list((await session.execute(edge_statement)).all())
        entity_documents = list(
            (await session.execute(entity_document_statement)).all()
        )

    graph_quality = compute_graph_quality(
        edges=edges,
        mention_count=int(totals["resolved_mentions"]),
        entity_count=int(totals["entities"]),
        entity_documents=entity_documents,
        evidence_count=int(totals["evidence_links"]),
    )

    return assemble_knowledge_statistics(
        totals=totals,
        source_counts=source_counts,
        job_counts=job_counts,
        pass_counts=pass_counts,
        entity_type_counts=entity_type_counts,
        graph_quality=graph_quality,
    )
