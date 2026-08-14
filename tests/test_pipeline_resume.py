from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from medical_kg.landing.loader import DocumentLoader


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
