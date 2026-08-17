import asyncio
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from medical_kg.config import DatabaseSettings
from medical_kg.db.models import Base, ExtractionChunk, ExtractionRun, RawAssertion
from medical_kg.db.repository import ExtractionRunSpec, KnowledgeRepository
from medical_kg.landing.chunking import chunk_document
from medical_kg.llm.base import LLMResponse
from medical_kg.models.assertion import ExtractionOutput
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
        "entity_resolutions",
        "raw_assertions",
        "assertions",
        "assertion_evidence",
        "relation_types",
        "invalid_records",
    }
    assert expected <= set(Base.metadata.tables)
    assert {"model_provider", "model_name", "temperature"} <= set(
        ExtractionChunk.__table__.columns
    )


@pytest.mark.asyncio
async def test_create_schema_adds_nullable_model_metadata_to_legacy_chunk_table(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database.as_posix()}")
    repository = KnowledgeRepository(engine)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TABLE extraction_chunk_jobs ("
                    "chunk_job_id VARCHAR(32) PRIMARY KEY, "
                    "status VARCHAR(16) NOT NULL)"
                )
            )

        await repository.create_schema()

        async with engine.connect() as connection:
            columns = {
                row[1]
                for row in (
                    await connection.execute(text("PRAGMA table_info(extraction_chunk_jobs)"))
                ).all()
            }
        assert {"model_provider", "model_name", "temperature"} <= columns
    finally:
        await engine.dispose()


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
        chunks = chunk_document(job.content)
        assert await repository.prepare_chunks(
            job=job,
            pass_name="general",
            chunks=chunks,
        ) == {}
        await repository.start_chunk(job.job_id, 0, job.worker_id)
        await repository.complete_chunk(
            job.job_id,
            0,
            LLMResponse(output=ExtractionOutput(), raw_output={"ok": True}),
            model_provider="provider-a",
            model_name="model-a",
            temperature=0.25,
        )
        stored = await repository.prepare_chunks(
            job=job,
            pass_name="general",
            chunks=chunks,
        )
        assert stored[0].model_provider == "provider-a"
        assert stored[0].model_name == "model-a"
        assert stored[0].temperature == 0.25

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


@pytest.mark.asyncio
async def test_sqlite_writers_wait_before_checking_out_another_connection(
    tmp_path: Path,
) -> None:
    database = tmp_path / "serialized.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{database.as_posix()}",
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
    )
    repository = KnowledgeRepository(engine)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first_writer() -> None:
        async with repository._write_session():
            first_entered.set()
            await release_first.wait()

    async def second_writer() -> None:
        async with repository._write_session():
            second_entered.set()

    first = asyncio.create_task(first_writer())
    await first_entered.wait()
    second = asyncio.create_task(second_writer())
    await asyncio.sleep(0.1)

    assert repository._write_lock.locked()
    assert not second_entered.is_set()

    release_first.set()
    await asyncio.gather(first, second)
    assert second_entered.is_set()
    await engine.dispose()


@pytest.mark.asyncio
async def test_complete_extraction_stores_raw_response_once_per_run(tmp_path: Path) -> None:
    database = tmp_path / "deduplicated.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database.as_posix()}")
    repository = KnowledgeRepository(engine)
    raw_output = {"provider_response": "large response" * 1000}
    try:
        await repository.create_schema()
        document = DocumentInput.from_content(
            document_id="doc-dedup",
            file_path=tmp_path / "doc.txt",
            content="A affects B. A treats C.",
        )
        await repository.register_document(document)
        await repository.enqueue_jobs(
            document_ids=[document.document_id],
            stages=["extract:general"],
            stage_version="extract:v1",
        )
        job = await repository.claim_job(
            stages=["extract:general"],
            stage_version="extract:v1",
            worker_id="test-worker",
        )
        assert job is not None
        output = ExtractionOutput.model_validate(
            {
                "assertions": [
                    {
                        "subject": {"mention": "A", "entity_type": "GENE"},
                        "object": {"mention": "B", "entity_type": "PROTEIN"},
                        "detailed_relation": "affects",
                        "evidence_text": "A affects B.",
                        "llm_confidence": 0.9,
                    },
                    {
                        "subject": {"mention": "A", "entity_type": "GENE"},
                        "object": {"mention": "C", "entity_type": "DISEASE"},
                        "detailed_relation": "treats",
                        "evidence_text": "A treats C.",
                        "llm_confidence": 0.8,
                    },
                ]
            }
        )
        await repository.complete_extraction(
            job=job,
            run_spec=ExtractionRunSpec(
                model_provider="test",
                model_name="test-model",
                prompt_name="test-prompt",
                prompt_version="v1",
                pass_name="general",
                temperature=0.0,
                code_version="test",
            ),
            output=output,
            raw_output=raw_output,
        )

        async with repository.sessions() as session:
            run_outputs = list(await session.scalars(select(ExtractionRun.raw_llm_output)))
            assertion_count = await session.scalar(
                select(func.count()).select_from(RawAssertion)
            )
        assert run_outputs == [raw_output]
        assert assertion_count == 2
        assert "raw_llm_output" not in RawAssertion.__table__.columns
    finally:
        await engine.dispose()
