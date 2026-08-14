import asyncio
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from medical_kg.config import DatabaseSettings
from medical_kg.db.models import Base
from medical_kg.db.repository import KnowledgeRepository
from medical_kg.landing.chunking import chunk_document
from medical_kg.models.document import DocumentInput


def test_required_sqlite_tables_are_declared() -> None:
    expected = {
        "documents",
        "document_revisions",
        "processing_jobs",
        "extraction_chunk_jobs",
        "api_rate_limits",
        "extraction_runs",
        "entity_mentions",
        "entities",
        "entity_aliases",
        "entity_external_ids",
        "raw_assertions",
        "assertions",
        "relation_types",
        "invalid_records",
    }
    assert expected <= set(Base.metadata.tables)


def test_database_settings_build_an_async_sqlite_url(tmp_path: Path) -> None:
    settings = DatabaseSettings(path=tmp_path / "knowledge.sqlite3")

    assert settings.url.startswith("sqlite+aiosqlite:///")
    assert settings.url.endswith("knowledge.sqlite3")


@pytest.mark.asyncio
async def test_sqlite_repository_lifecycle(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database.as_posix()}")
    repository = KnowledgeRepository(engine)
    try:
        await repository.create_schema()
        assert await repository.seed_relations(["TREATS"]) == 1
        assert await repository.seed_relations(["TREATS"]) == 0

        document = DocumentInput.from_content(
            document_id="doc-1",
            file_path=tmp_path / "doc.txt",
            content="Metformin treats diabetes.",
        )
        assert await repository.register_document(document) == (True, True)
        second_document = DocumentInput.from_content(
            document_id="doc-2",
            file_path=tmp_path / "doc-2.txt",
            content="Insulin lowers blood glucose.",
        )
        assert await repository.register_document(second_document) == (True, True)
        assert await repository.enqueue_jobs(
            document_ids=[document.document_id, second_document.document_id],
            stages=["extract:general"],
            stage_version="extract:v1",
        ) == 2

        jobs = await asyncio.gather(
            repository.claim_job(
                stages=["extract:general"],
                stage_version="extract:v1",
                worker_id="test-worker-1",
            ),
            repository.claim_job(
                stages=["extract:general"],
                stage_version="extract:v1",
                worker_id="test-worker-2",
            ),
        )
        assert all(job is not None for job in jobs)
        assert {job.document_id for job in jobs if job is not None} == {"doc-1", "doc-2"}
        job = jobs[0]
        assert job is not None
        assert await repository.prepare_chunks(
            job=job,
            pass_name="general",
            chunks=chunk_document(job.content),
        ) == {}

        assert await repository.reserve_api_capacity(
            limiter_key="provider:model",
            estimated_tokens=10,
            requests_per_minute=100,
            tokens_per_minute=10_000,
        ) == 0
        assert await repository.reserve_api_capacity(
            limiter_key="provider:model",
            estimated_tokens=10,
            requests_per_minute=100,
            tokens_per_minute=10_000,
        ) > 0
    finally:
        await engine.dispose()

    assert database.is_file()
