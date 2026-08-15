import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest

from medical_kg.bronze.extraction import BronzeExtractor
from medical_kg.db.models import (
    Assertion,
    Document,
    Entity,
    EntityMention,
    ExtractionRun,
    RawAssertion,
)
from medical_kg.db.repository import ClaimedJob, StoredChunkResult
from medical_kg.landing.chunking import chunk_document
from medical_kg.landing.loader import DocumentLoader
from medical_kg.llm.base import LLMResponse
from medical_kg.models.assertion import ExtractionOutput
from medical_kg.models.source import SourceType, merge_sources
from medical_kg.prompts import PromptDefinition, PromptRegistry


def test_source_lists_are_declared_on_graph_and_bronze_records() -> None:
    assert "source_type" in Document.__table__.columns
    assert "source_type" in ExtractionRun.__table__.columns
    for model in (EntityMention, RawAssertion, Entity, Assertion):
        assert "sources" in model.__table__.columns


def test_merge_sources_keeps_multiple_unique_contributions() -> None:
    sources = merge_sources(
        [{"document_id": "book-1", "source_type": "textbook"}],
        [
            {"document_id": "book-1", "source_type": "textbook"},
            {"document_id": "guide-1", "source_type": "guidelines"},
        ],
    )

    assert sources == [
        {"document_id": "book-1", "source_type": "textbook"},
        {"document_id": "guide-1", "source_type": "guidelines"},
    ]


def test_chunking_is_opt_in_and_supports_overlap() -> None:
    unchunked = chunk_document("abcdefghij")
    chunked = chunk_document("abcdefghij", chunk_size=5, chunk_overlap=1)

    assert [chunk.content for chunk in unchunked] == ["abcdefghij"]
    assert [(chunk.character_start, chunk.character_end, chunk.content) for chunk in chunked] == [
        (0, 5, "abcde"),
        (4, 9, "efghi"),
        (8, 10, "ij"),
    ]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(None, 1), (5, -1), (5, 5)],
)
def test_invalid_chunking_options_are_rejected(chunk_size: int | None, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document("text", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_json_source_type_overrides_ingest_default(tmp_path: Path) -> None:
    path = tmp_path / "guide.json"
    path.write_text(
        '{"content":"Recommendation text.","source_type":"guidelines"}',
        encoding="utf-8",
    )

    document = DocumentLoader(AsyncMock()).load_file(
        path, tmp_path, default_source_type=SourceType.TEXTBOOK
    )

    assert document.source_type == SourceType.GUIDELINES


@pytest.mark.asyncio
async def test_extractor_processes_each_chunk_and_persists_one_merged_run() -> None:
    settings = SimpleNamespace(
        processing=SimpleNamespace(
            requests_per_minute=100,
            tokens_per_minute=100_000,
            reserved_output_tokens=0,
            distributed_rate_limit=False,
            api_concurrency=100,
            chunk_queue_size=300,
            max_retries=0,
            retry_backoff=1,
            request_timeout=10,
        ),
        extraction=SimpleNamespace(
            stage_version="extract:v1", passes=["general"], code_version="x"
        ),
        llm=SimpleNamespace(provider="test", model="test-model", temperature=0.0),
    )
    repository = AsyncMock()
    repository.prepare_chunks.return_value = {}
    repository.complete_extraction.return_value = uuid4()
    llm = AsyncMock()
    llm.extract_document.side_effect = [
        LLMResponse(output=ExtractionOutput(), raw_output={"chunk": index}) for index in range(3)
    ]
    prompts = SimpleNamespace(
        extraction=Mock(
            return_value=PromptDefinition(
                name="test", version="v1", system_prompt="system", user_template="{document}"
            )
        )
    )
    extractor = BronzeExtractor(settings=settings, repository=repository, llm=llm, prompts=prompts)
    job = ClaimedJob(
        job_id=uuid4(),
        document_id="doc-1",
        stage="extract:general",
        stage_version="extract:v1|chunk=5,overlap=1",
        retry_count=0,
        content="abcdefghij",
        content_hash="0" * 64,
        source_type="research",
    )

    result = await extractor._process(job, chunk_size=5, chunk_overlap=1)

    assert result.documents_successful == 1
    assert result.requests == 3
    assert [call.kwargs["user_prompt"] for call in llm.extract_document.await_args_list] == [
        "abcde",
        "efghi",
        "ij",
    ]
    persisted = repository.complete_extraction.await_args.kwargs
    assert persisted["raw_output"]["chunking"] == {"chunk_size": 5, "chunk_overlap": 1}
    assert len(persisted["raw_output"]["chunks"]) == 3


@pytest.mark.asyncio
async def test_extractor_retries_without_an_external_retry_framework() -> None:
    settings = SimpleNamespace(
        processing=SimpleNamespace(
            requests_per_minute=100,
            tokens_per_minute=100_000,
            reserved_output_tokens=0,
            distributed_rate_limit=False,
            max_retries=1,
            retry_backoff=0,
            request_timeout=10,
        ),
        llm=SimpleNamespace(temperature=0.0),
    )
    llm = AsyncMock()
    llm.extract_document.side_effect = [
        httpx.ReadTimeout("temporary failure"),
        LLMResponse(output=ExtractionOutput(), raw_output={}),
    ]
    extractor = BronzeExtractor(
        settings=settings,
        repository=AsyncMock(),
        llm=llm,
        prompts=AsyncMock(),
    )
    statistics = SimpleNamespace(requests=0)

    response = await extractor._request("text", "system", "user", statistics)

    assert response.output.assertions == []
    assert statistics.requests == 2


@pytest.mark.asyncio
async def test_extractor_does_not_retry_permanent_http_errors() -> None:
    settings = SimpleNamespace(
        processing=SimpleNamespace(
            requests_per_minute=100,
            tokens_per_minute=100_000,
            reserved_output_tokens=0,
            distributed_rate_limit=False,
            max_retries=4,
            retry_backoff=1,
            request_timeout=10,
        ),
        llm=SimpleNamespace(temperature=0.0),
    )
    request = httpx.Request("POST", "https://example.test/chat/completions")
    response = httpx.Response(400, request=request)
    error = httpx.HTTPStatusError("bad request", request=request, response=response)
    llm = AsyncMock()
    llm.extract_document.side_effect = error
    extractor = BronzeExtractor(
        settings=settings,
        repository=AsyncMock(),
        llm=llm,
        prompts=AsyncMock(),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await extractor._request("text", "system", "user", SimpleNamespace(requests=0))

    assert llm.extract_document.await_count == 1


@pytest.mark.asyncio
async def test_successful_chunk_checkpoint_is_reused_without_an_api_request() -> None:
    settings = SimpleNamespace(
        processing=SimpleNamespace(
            requests_per_minute=100_000,
            tokens_per_minute=100_000_000,
            reserved_output_tokens=0,
            distributed_rate_limit=False,
            api_concurrency=100,
            chunk_queue_size=300,
            max_retries=0,
            retry_backoff=1,
            request_timeout=10,
        ),
        extraction=SimpleNamespace(
            stage_version="extract:v1", passes=["general"], code_version="x"
        ),
        llm=SimpleNamespace(provider="test", model="test-model", temperature=0.0),
    )
    repository = AsyncMock()
    repository.prepare_chunks.return_value = {
        0: StoredChunkResult(
            chunk_index=0,
            validated_output={"assertions": []},
            raw_output={"checkpoint": 0},
            input_tokens=10,
            output_tokens=2,
        )
    }
    repository.complete_extraction.return_value = uuid4()
    llm = AsyncMock()
    llm.extract_document.return_value = LLMResponse(
        output=ExtractionOutput(), raw_output={"requested": 1}
    )
    prompts = SimpleNamespace(
        extraction=Mock(
            return_value=PromptDefinition(
                name="test", version="v1", system_prompt="system", user_template="{document}"
            )
        )
    )
    extractor = BronzeExtractor(settings=settings, repository=repository, llm=llm, prompts=prompts)
    job = ClaimedJob(
        job_id=uuid4(),
        document_id="doc-resume",
        stage="extract:general",
        stage_version="extract:v1|chunk=1,overlap=0",
        retry_count=1,
        content="ab",
        content_hash="0" * 64,
        source_type="research",
    )

    result = await extractor._process(job, chunk_size=1)

    assert result.documents_successful == 1
    assert result.requests == 1
    assert llm.extract_document.await_count == 1
    repository.start_chunk.assert_awaited_once_with(job.job_id, 1, "")


@pytest.mark.asyncio
async def test_single_chunked_job_can_reach_one_hundred_api_requests_in_flight() -> None:
    settings = SimpleNamespace(
        processing=SimpleNamespace(
            requests_per_minute=10_000_000,
            tokens_per_minute=1_000_000_000,
            reserved_output_tokens=0,
            distributed_rate_limit=False,
            api_concurrency=100,
            chunk_queue_size=300,
            max_retries=0,
            retry_backoff=0,
            request_timeout=10,
        ),
        extraction=SimpleNamespace(
            stage_version="extract:v1", passes=["general"], code_version="x"
        ),
        llm=SimpleNamespace(provider="test", model="test-model", temperature=0.0),
    )
    repository = AsyncMock()
    repository.prepare_chunks.return_value = {}
    repository.complete_extraction.return_value = uuid4()
    active = 0
    peak = 0
    reached_one_hundred = asyncio.Event()
    release = asyncio.Event()

    async def extract_document(**_: object) -> LLMResponse:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if peak == 100:
            reached_one_hundred.set()
        await release.wait()
        active -= 1
        return LLMResponse(output=ExtractionOutput(), raw_output={})

    llm = AsyncMock()
    llm.extract_document.side_effect = extract_document
    prompts = SimpleNamespace(
        extraction=Mock(
            return_value=PromptDefinition(
                name="test", version="v1", system_prompt="system", user_template="{document}"
            )
        )
    )
    extractor = BronzeExtractor(settings=settings, repository=repository, llm=llm, prompts=prompts)
    job = ClaimedJob(
        job_id=uuid4(),
        document_id="doc-100",
        stage="extract:general",
        stage_version="extract:v1|chunk=1,overlap=0",
        retry_count=0,
        content="x" * 120,
        content_hash="0" * 64,
        source_type="research",
    )

    processing = asyncio.create_task(extractor._process(job, chunk_size=1))
    await asyncio.wait_for(reached_one_hundred.wait(), timeout=3)
    assert peak == 100
    release.set()
    result = await processing

    assert result.documents_successful == 1
    assert result.requests == 120


def test_chunk_settings_are_part_of_resumable_stage_version() -> None:
    extractor = object.__new__(BronzeExtractor)
    extractor.settings = SimpleNamespace(
        extraction=SimpleNamespace(stage_version="extract:v1", passes=["general"])
    )
    extractor.prompts = SimpleNamespace(
        extraction=Mock(return_value=SimpleNamespace(version="v2"))
    )

    assert extractor.stage_version(None, 0) == "extract:v1|prompts=general:v2"
    assert (
        extractor.stage_version(5000, 200)
        == "extract:v1|prompts=general:v2|chunk=5000,overlap=200"
    )


def test_v3_extraction_prompts_state_strict_output_rules() -> None:
    registry = PromptRegistry(Path(__file__).parents[1] / "prompts")

    for pass_name in ("general", "molecular", "clinical"):
        prompt = registry.extraction(pass_name)
        assert prompt.version == "v3"
        assert "entity_type_detail" in prompt.system_prompt
        assert "never return null" in prompt.system_prompt


@pytest.mark.asyncio
async def test_internal_correction_request_is_included_in_run_statistics() -> None:
    settings = SimpleNamespace(
        processing=SimpleNamespace(
            requests_per_minute=100_000,
            tokens_per_minute=100_000_000,
            reserved_output_tokens=0,
            distributed_rate_limit=False,
            max_retries=0,
            retry_backoff=0,
            request_timeout=10,
        ),
        llm=SimpleNamespace(temperature=0.0),
    )
    llm = AsyncMock()
    llm.extract_document.return_value = LLMResponse(
        output=ExtractionOutput(),
        raw_output={},
        input_tokens=20,
        output_tokens=5,
        metadata={"request_count": 2},
    )
    extractor = BronzeExtractor(
        settings=settings,
        repository=AsyncMock(),
        llm=llm,
        prompts=AsyncMock(),
    )
    statistics = SimpleNamespace(requests=0)

    await extractor._request("text", "system", "user", statistics)

    assert statistics.requests == 2
