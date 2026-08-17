from pathlib import Path

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import create_async_engine

from medical_kg.db.models import (
    Assertion,
    AssertionEvidence,
    Entity,
    EntityResolution,
    RawAssertion,
    RelationType,
)
from medical_kg.db.repository import ExtractionRunSpec, KnowledgeRepository
from medical_kg.models.assertion import ExtractionOutput
from medical_kg.models.document import DocumentInput
from medical_kg.prompts import PromptRegistry
from medical_kg.silver.canonicalization import CanonicalizationPipeline


async def _store_extraction(
    repository: KnowledgeRepository,
    tmp_path: Path,
    *,
    document_id: str,
    content: str,
    subject: str,
    object_: str,
    relation: str,
) -> None:
    document = DocumentInput.from_content(
        document_id=document_id,
        file_path=tmp_path / f"{document_id}.txt",
        content=content,
    )
    await repository.register_document(document)
    await repository.enqueue_jobs(
        document_ids=[document_id],
        stages=["extract:general"],
        stage_version="extract:test",
    )
    job = await repository.claim_job(
        stages=["extract:general"],
        stage_version="extract:test",
        worker_id="test-worker",
        document_id=document_id,
    )
    assert job is not None
    output = ExtractionOutput.model_validate(
        {
            "assertions": [
                {
                    "subject": {"mention": subject, "entity_type": "DISEASE"},
                    "object": {"mention": object_, "entity_type": "DRUG"},
                    "detailed_relation": relation,
                    "evidence_text": content,
                    "llm_confidence": 0.95,
                }
            ]
        }
    )
    await repository.complete_extraction(
        job=job,
        run_spec=ExtractionRunSpec(
            model_provider="test",
            model_name="test",
            prompt_name="test",
            prompt_version="v1",
            pass_name="general",
            temperature=0.0,
            code_version="test",
        ),
        output=output,
        raw_output={},
    )


@pytest.mark.asyncio
async def test_canonicalization_resolves_aliases_and_aggregates_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'canonical.sqlite3').as_posix()}"
    )
    repository = KnowledgeRepository(engine)
    try:
        await repository.create_schema()
        await _store_extraction(
            repository,
            tmp_path,
            document_id="doc-1",
            content="T2DM is treated with metformin.",
            subject="T2DM",
            object_="metformin",
            relation="is treated with",
        )
        await _store_extraction(
            repository,
            tmp_path,
            document_id="doc-2",
            content="Type 2 diabetes mellitus was treated by Metformin.",
            subject="Type 2 diabetes mellitus",
            object_="Metformin",
            relation="was treated by",
        )
        pipeline = CanonicalizationPipeline(
            repository=repository,
            vocabulary=["treats", "associated_with", "OTHER"],
            prompts=PromptRegistry(Path(__file__).parents[1] / "prompts"),
        )

        def reject_fuzzy_retrieval(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("non-semantic canonicalization must skip fuzzy retrieval")

        monkeypatch.setattr(pipeline.retriever, "retrieve", reject_fuzzy_retrieval)

        progress_events: list[tuple[str, int, int, str]] = []
        first = await pipeline.run(progress=lambda *event: progress_events.append(event))
        second = await pipeline.run()

        assert first.entities_created == 2
        assert first.canonical_assertions_created == 1
        assert first.evidence_links_created == 2
        assert first.duplicate_assertions_aggregated == 1
        assert second.entities_created == 0
        assert second.canonical_assertions_created == 0
        assert second.evidence_links_created == 0
        completed_phases = {
            phase
            for phase, completed, total, _unit in progress_events
            if completed == total
        }
        assert completed_phases == {
            "Preparing database",
            "Loading canonicalization data",
            "Resolving entity mentions",
            "Canonicalizing assertions",
        }

        async with repository.sessions() as session:
            counts = {
                "entities": await session.scalar(select(func.count()).select_from(Entity)),
                "resolutions": await session.scalar(
                    select(func.count()).select_from(EntityResolution)
                ),
                "raw": await session.scalar(
                    select(func.count()).select_from(RawAssertion)
                ),
                "assertions": await session.scalar(
                    select(func.count()).select_from(Assertion)
                ),
                "evidence": await session.scalar(
                    select(func.count()).select_from(AssertionEvidence)
                ),
            }
            relation = await session.scalar(
                select(RelationType.canonical_name)
                .join(
                    Assertion,
                    Assertion.canonical_relation_id == RelationType.relation_id,
                )
                .limit(1)
            )
            disease_name = await session.scalar(
                select(Entity.canonical_name).where(Entity.entity_type == "DISEASE")
            )
        assert counts == {
            "entities": 2,
            "resolutions": 4,
            "raw": 2,
            "assertions": 1,
            "evidence": 2,
        }
        assert relation == "treats"
        assert disease_name == "type 2 diabetes mellitus"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_canonicalization_resumes_after_resolution_batch_checkpoint(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'resolution-resume.sqlite3').as_posix()}"
    )
    repository = KnowledgeRepository(engine)
    try:
        await repository.create_schema()
        for index in range(3):
            await _store_extraction(
                repository,
                tmp_path,
                document_id=f"resume-{index}",
                content=f"Disease {index} is treated with drug {index}.",
                subject=f"Disease {index}",
                object_=f"drug {index}",
                relation="is treated with",
            )
        pipeline = CanonicalizationPipeline(
            repository=repository,
            vocabulary=["treats", "OTHER"],
            prompts=PromptRegistry(Path(__file__).parents[1] / "prompts"),
            batch_size=2,
        )

        def interrupt_after_first_batch(
            phase: str, completed: int, _total: int, _unit: str
        ) -> None:
            if phase == "Resolving entity mentions" and completed == 2:
                raise InterruptedError("simulated Ctrl+C")

        with pytest.raises(InterruptedError, match=r"simulated Ctrl\+C"):
            await pipeline.run(progress=interrupt_after_first_batch)

        async with repository.sessions() as session:
            checkpointed = await session.scalar(
                select(func.count()).select_from(EntityResolution)
            )
        assert checkpointed == 2

        resumed = await pipeline.run()
        assert resumed.mentions_resolved == 6
        async with repository.sessions() as session:
            resolutions = await session.scalar(
                select(func.count()).select_from(EntityResolution)
            )
            evidence = await session.scalar(
                select(func.count()).select_from(AssertionEvidence)
            )
        assert resolutions == 6
        assert evidence == 3
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_canonicalization_resumes_after_assertion_batch_checkpoint(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'assertion-resume.sqlite3').as_posix()}"
    )
    repository = KnowledgeRepository(engine)
    try:
        await repository.create_schema()
        for index in range(2):
            await _store_extraction(
                repository,
                tmp_path,
                document_id=f"fact-{index}",
                content=f"Disease {index} is treated with drug {index}.",
                subject=f"Disease {index}",
                object_=f"drug {index}",
                relation="is treated with",
            )
        pipeline = CanonicalizationPipeline(
            repository=repository,
            vocabulary=["treats", "OTHER"],
            prompts=PromptRegistry(Path(__file__).parents[1] / "prompts"),
            batch_size=1,
        )

        def interrupt_after_first_batch(
            phase: str, completed: int, _total: int, _unit: str
        ) -> None:
            if phase == "Canonicalizing assertions" and completed == 1:
                raise InterruptedError("simulated Ctrl+C")

        with pytest.raises(InterruptedError, match=r"simulated Ctrl\+C"):
            await pipeline.run(progress=interrupt_after_first_batch)

        async with repository.sessions() as session:
            checkpointed_assertions = await session.scalar(
                select(func.count()).select_from(Assertion)
            )
            checkpointed_evidence = await session.scalar(
                select(func.count()).select_from(AssertionEvidence)
            )
        assert checkpointed_assertions == 1
        assert checkpointed_evidence == 1

        await pipeline.run()
        async with repository.sessions() as session:
            assertions = await session.scalar(select(func.count()).select_from(Assertion))
            evidence = await session.scalar(
                select(func.count()).select_from(AssertionEvidence)
            )
        assert assertions == 2
        assert evidence == 2
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_snapshot_queries_do_not_expand_mention_ids_into_parameters(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'snapshot.sqlite3').as_posix()}"
    )
    repository = KnowledgeRepository(engine)
    try:
        await repository.create_schema()
        for index in range(5):
            await _store_extraction(
                repository,
                tmp_path,
                document_id=f"doc-{index}",
                content=f"Disease {index} is treated with drug {index}.",
                subject=f"Disease {index}",
                object_=f"drug {index}",
                relation="is treated with",
            )

        @event.listens_for(engine.sync_engine, "before_cursor_execute")
        def reject_large_parameter_lists(
            _connection: object,
            _cursor: object,
            _statement: str,
            parameters: object,
            _context: object,
            executemany: bool,
        ) -> None:
            if not executemany and isinstance(parameters, (list, tuple)):
                assert len(parameters) <= 8

        pipeline = CanonicalizationPipeline(
            repository=repository,
            vocabulary=["treats", "OTHER"],
            prompts=PromptRegistry(Path(__file__).parents[1] / "prompts"),
        )
        snapshot = await pipeline._load_snapshot(document_id=None)
        filtered = await pipeline._load_snapshot(document_id="doc-0")

        assert len(snapshot["raw_assertions"]) == 5
        assert len(snapshot["mentions"]) == 10
        assert snapshot["documents"] == {}
        assert len(filtered["raw_assertions"]) == 1
        assert len(filtered["mentions"]) == 2
        assert filtered["documents"] == {}
    finally:
        await engine.dispose()
