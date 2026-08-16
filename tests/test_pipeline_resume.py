from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from medical_kg.landing.loader import DocumentLoader
from medical_kg.pipeline.runner import PipelineRunner


@pytest.mark.asyncio
async def test_unchanged_document_is_registered_without_forcing_a_new_job(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text("A complete paper.", encoding="utf-8")
    repository = AsyncMock()
    repository.register_document.return_value = (False, False)

    result = await DocumentLoader(repository).ingest(source)

    assert result.unchanged == 1
    assert result.changed == 0
    repository.register_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_changed_document_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text("Updated paper.", encoding="utf-8")
    repository = AsyncMock()
    repository.register_document.return_value = (False, True)

    result = await DocumentLoader(repository).ingest(source)

    assert result.changed == 1


@pytest.mark.asyncio
async def test_ingest_reports_human_progress(tmp_path: Path) -> None:
    source = tmp_path / "papers"
    source.mkdir()
    (source / "one.txt").write_text("First paper.", encoding="utf-8")
    (source / "two.txt").write_text("Second paper.", encoding="utf-8")
    repository = AsyncMock()
    repository.register_document.return_value = (True, True)
    updates: list[int] = []
    loader = DocumentLoader(repository)

    assert loader.count(source) == 2
    await loader.ingest(source, progress=lambda result: updates.append(result.discovered))

    assert updates == [1, 2]


@pytest.mark.asyncio
async def test_extraction_reports_actual_pending_job_total() -> None:
    repository = AsyncMock()
    repository.count_pending_jobs.return_value = 3
    extractor = AsyncMock()
    extractor.settings = SimpleNamespace(
        processing=SimpleNamespace(request_timeout=1, max_retries=0)
    )
    extractor.stage_version = Mock(return_value="extract:v1")
    extractor.job_stages = ["extract:general"]
    totals: list[int] = []
    runner = PipelineRunner(
        repository=repository,
        loader=DocumentLoader(repository),
        extractor=extractor,
    )

    await runner.extract(document_ids=["doc-1"], progress_total=totals.append)

    assert totals == [3]
    repository.count_pending_jobs.assert_awaited_once_with(
        stages=["extract:general"],
        stage_version="extract:v1",
        document_id=None,
    )
