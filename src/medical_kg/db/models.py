from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UTCDateTime(TypeDecorator[datetime]):
    """Store UTC timestamps in SQLite and restore their timezone on read."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now(), nullable=False
    )


class UpdatedTimestampMixin(TimestampMixin):
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Document(Base, UpdatedTimestampMixin):
    __tablename__ = "documents"
    document_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(String(255))
    pmid: Mapped[str | None] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="research", server_default="research"
    )
    __table_args__ = (
        CheckConstraint(
            "source_type IN ('textbook', 'guidelines', 'research')",
            name="ck_document_source_type",
        ),
        Index("ix_documents_content_hash", "content_hash"),
        Index("ix_documents_doi", "doi"),
        Index("ix_documents_pmid", "pmid"),
    )


class DocumentRevision(Base, TimestampMixin):
    """Immutable source snapshots keep earlier Bronze evidence reproducible."""

    __tablename__ = "document_revisions"
    revision_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="research", server_default="research"
    )
    __table_args__ = (
        UniqueConstraint("document_id", "content_hash", name="uq_document_revision"),
        Index("ix_document_revisions_hash", "content_hash"),
    )


class ExtractionRun(Base, TimestampMixin):
    __tablename__ = "extraction_runs"
    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    pass_name: Mapped[str] = mapped_column(String(128), nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    code_version: Mapped[str] = mapped_column(String(128), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Store the provider response exactly once per extraction pass.
    raw_llm_output: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(128), nullable=False)
    stage_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(255))
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    error_message: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        UniqueConstraint("document_id", "stage", "stage_version", name="uq_processing_unit"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED')", name="ck_job_status"
        ),
        Index("ix_jobs_status_stage", "status", "stage"),
    )


class ExtractionChunk(Base, UpdatedTimestampMixin):
    """Durable API-level checkpoint for one deterministic document chunk."""

    __tablename__ = "extraction_chunk_jobs"
    chunk_job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("processing_jobs.job_id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    pass_name: Mapped[str] = mapped_column(String(128), nullable=False)
    stage_version: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    character_start: Mapped[int] = mapped_column(Integer, nullable=False)
    character_end: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    worker_id: Mapped[str | None] = mapped_column(String(255))
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    error_message: Mapped[str | None] = mapped_column(Text)
    validated_output: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    raw_output: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "pass_name",
            "stage_version",
            "content_hash",
            "chunk_index",
            name="uq_extraction_chunk_unit",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED')",
            name="ck_extraction_chunk_status",
        ),
        CheckConstraint("input_tokens >= 0", name="ck_extraction_chunk_input_tokens"),
        CheckConstraint("output_tokens >= 0", name="ck_extraction_chunk_output_tokens"),
        Index("ix_extraction_chunks_job_status", "job_id", "status"),
    )


class APIRateLimit(Base):
    """Shared virtual schedule used to enforce provider limits across processes."""

    __tablename__ = "api_rate_limits"
    limiter_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    next_request_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    next_token_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class EntityMention(Base, TimestampMixin):
    __tablename__ = "entity_mentions"
    mention_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extraction_runs.extraction_run_id"), nullable=False
    )
    mention_text: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type_detail: Mapped[str | None] = mapped_column(Text)
    section: Mapped[str | None] = mapped_column(Text)
    paragraph: Mapped[int | None] = mapped_column(Integer)
    sentence: Mapped[int | None] = mapped_column(Integer)
    character_start: Mapped[int | None] = mapped_column(Integer)
    character_end: Mapped[int | None] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    __table_args__ = (
        CheckConstraint("json_type(sources) = 'array'", name="ck_mention_sources_array"),
        Index("ix_mentions_document", "document_id"),
    )


class Entity(Base, UpdatedTimestampMixin):
    __tablename__ = "entities"
    entity_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    __table_args__ = (
        CheckConstraint("json_type(sources) = 'array'", name="ck_entity_sources_array"),
    )


class EntityAlias(Base, TimestampMixin):
    __tablename__ = "entity_aliases"
    alias_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.entity_id", ondelete="CASCADE"), nullable=False
    )
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    alias_source: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (
        UniqueConstraint("entity_id", "alias", name="uq_entity_alias"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_alias_confidence",
        ),
    )


class EntityExternalId(Base, TimestampMixin):
    __tablename__ = "entity_external_ids"
    mapping_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.entity_id", ondelete="CASCADE"), nullable=False
    )
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    accession: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_id: Mapped[str] = mapped_column(String(255), nullable=False)
    mapping_method: Mapped[str] = mapped_column(String(128), nullable=False)
    mapping_source: Mapped[str] = mapped_column(String(255), nullable=False)
    mapping_confidence: Mapped[float | None] = mapped_column(Float)
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    __table_args__ = (
        UniqueConstraint("entity_id", "namespace", "accession", name="uq_external_mapping"),
        CheckConstraint(
            "mapping_confidence IS NULL OR (mapping_confidence >= 0 AND mapping_confidence <= 1)",
            name="ck_mapping_confidence",
        ),
    )


class RawAssertion(Base, TimestampMixin):
    __tablename__ = "raw_assertions"
    raw_assertion_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extraction_runs.extraction_run_id"), nullable=False
    )
    subject_mention_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity_mentions.mention_id"), nullable=False
    )
    object_mention_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity_mentions.mention_id"), nullable=False
    )
    subject_mention: Mapped[str] = mapped_column(Text, nullable=False)
    subject_type: Mapped[str] = mapped_column(String(64), nullable=False)
    object_mention: Mapped[str] = mapped_column(Text, nullable=False)
    object_type: Mapped[str] = mapped_column(String(64), nullable=False)
    detailed_relation: Mapped[str] = mapped_column(Text, nullable=False)
    llm_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    qualifiers: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    negated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    speculative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    evidence_validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    validation_error: Mapped[str | None] = mapped_column(Text)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    __table_args__ = (
        CheckConstraint("llm_confidence >= 0 AND llm_confidence <= 1", name="ck_llm_confidence"),
        CheckConstraint("json_type(sources) = 'array'", name="ck_raw_assertion_sources_array"),
        Index("ix_raw_assertions_document", "document_id"),
    )


class RelationType(Base, UpdatedTimestampMixin):
    __tablename__ = "relation_types"
    relation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    canonical_name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    parent_relation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("relation_types.relation_id")
    )
    inverse_relation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("relation_types.relation_id")
    )
    deprecated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class Assertion(Base, UpdatedTimestampMixin):
    __tablename__ = "assertions"
    assertion_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    raw_assertion_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("raw_assertions.raw_assertion_id"), nullable=False, unique=True
    )
    subject_entity_id: Mapped[str] = mapped_column(ForeignKey("entities.entity_id"), nullable=False)
    object_entity_id: Mapped[str] = mapped_column(ForeignKey("entities.entity_id"), nullable=False)
    canonical_relation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("relation_types.relation_id"), nullable=False
    )
    qualifiers: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    negated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    speculative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    normalized_identity: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    __table_args__ = (
        CheckConstraint("json_type(sources) = 'array'", name="ck_assertion_sources_array"),
    )


class InvalidRecord(Base, TimestampMixin):
    __tablename__ = "invalid_records"
    invalid_record_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.document_id"))
    stage: Mapped[str] = mapped_column(String(128), nullable=False)
    source_table: Mapped[str] = mapped_column(String(128), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String(255))
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
