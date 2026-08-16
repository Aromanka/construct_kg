from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from medical_kg.openalex.catalog import OpenAlexCatalog
from medical_kg.openalex.cli import _selection_options, build_parser
from medical_kg.openalex.filtering import WorkFilter
from medical_kg.openalex.fulltext import FullTextResolver
from medical_kg.openalex.models import OpenAlexWork, restore_abstract
from medical_kg.openalex.pipeline import OpenAlexPipeline, SelectionOptions
from medical_kg.openalex.snapshot import OpenAlexSnapshot


def _work(
    work_id: str,
    title: str,
    abstract: str,
    *,
    source: str = "Journal of Voice",
    fulltext: bool = True,
) -> dict[str, object]:
    positions = {token: [index] for index, token in enumerate(abstract.split())}
    return {
        "id": f"https://openalex.org/{work_id}",
        "doi": f"https://doi.org/10.1/{work_id.lower()}",
        "title": title,
        "abstract_inverted_index": positions,
        "publication_year": 2025,
        "language": "en",
        "type": "article",
        "has_content": {"grobid_xml": fulltext, "pdf": fulltext},
        "primary_location": {
            "source": {
                "id": "https://openalex.org/S42",
                "display_name": source,
                "type": "journal",
            }
        },
        "locations": [],
        "referenced_works": ["https://openalex.org/W999"],
    }


def _snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "openalex-snapshot"
    works = root / "data" / "works" / "updated_date=2025-01-01"
    sources = root / "data" / "sources" / "updated_date=2025-01-01"
    works.mkdir(parents=True)
    sources.mkdir(parents=True)
    records = [
        _work("W1", "Diabetes voice biomarkers", "voice changes in diabetes"),
        _work("W2", "Unrelated astronomy", "galaxy formation", fulltext=False),
        _work("W3", "Diabetes review", "systematic review of diabetes"),
    ]
    with gzip.open(works / "part_0000.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    with gzip.open(works / "part_0001.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(_work("W4", "Not manifested", "ignored copy")) + "\n")
    (works.parent / "manifest").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "url": (
                            "s3://openalex/data/works/updated_date=2025-01-01/part_0000.gz"
                        )
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    # Interrupted-copy suffix must not be discovered.
    (works / "part_0000.gz.BAD123").write_bytes(b"not gzip")
    source = {
        "id": "https://openalex.org/S42",
        "display_name": "Journal of Voice",
        "issn_l": "1234-5678",
    }
    with gzip.open(sources / "part_0000.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(source) + "\n")
    return root


class _FakeScreener:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def screen(self, works: list[OpenAlexWork], instruction: str) -> set[int]:
        self.batch_sizes.append(len(works))
        assert instruction == "exclude reviews"
        return {index for index, work in enumerate(works) if "review" not in work.title.lower()}

    async def aclose(self) -> None:
        return None


def test_restore_abstract_uses_positions() -> None:
    assert restore_abstract({"world": [1], "hello": [0, 2]}) == "hello world hello"
    assert restore_abstract(None) is None


def test_cli_requires_an_intentional_filter_and_treats_lone_id_as_explicit() -> None:
    parser = build_parser()
    unfiltered = parser.parse_args(["select", "snapshot"])
    with pytest.raises(ValueError, match="At least one filter"):
        _selection_options(unfiltered)

    explicit = parser.parse_args(["select", "snapshot", "--include-id", "W1"])
    assert _selection_options(explicit).max_candidates == 0


def test_snapshot_strictly_discovers_gzip_parts(tmp_path: Path) -> None:
    snapshot = OpenAlexSnapshot(_snapshot(tmp_path))

    assert [path.name for path in snapshot.entity_files("works")] == ["part_0000.gz"]
    assert [work.work_id for work in snapshot.iter_works()] == ["W1", "W2", "W3"]


@pytest.mark.asyncio
async def test_select_batches_llm_and_retains_stable_locator(tmp_path: Path) -> None:
    snapshot = OpenAlexSnapshot(_snapshot(tmp_path))
    workspace = tmp_path / "feature"
    screener = _FakeScreener()
    with OpenAlexCatalog(workspace / "catalog.sqlite3") as catalog:
        pipeline = OpenAlexPipeline(snapshot=snapshot, catalog=catalog, workspace=workspace)
        statistics = await pipeline.select(
            SelectionOptions(
                work_filter=WorkFilter(
                    keywords=["diabetes"],
                    sources=["Journal of Voice"],
                    require_fulltext=True,
                ),
                llm_instruction="exclude reviews",
                llm_batch_size=2,
            ),
            screener=screener,
        )

        assert statistics.scanned == 3
        assert statistics.candidates == 2
        assert statistics.selected == 1
        assert screener.batch_sizes == [2]
        row = catalog.get("https://openalex.org/W1")
        assert row is not None
        assert row["document_id"] == "openalex:W1"
        assert row["snapshot_file"].endswith("part_0000.gz")
        assert row["snapshot_line"] == 1
        assert json.loads(row["raw_json"])["referenced_works"]
        assert pipeline.enrich_sources() == 1


@pytest.mark.asyncio
async def test_add_and_materialize_local_fulltext(tmp_path: Path) -> None:
    snapshot = OpenAlexSnapshot(_snapshot(tmp_path))
    workspace = tmp_path / "feature"
    local = tmp_path / "fulltext"
    local.mkdir()
    (local / "W2.txt").write_text("Complete article body.", encoding="utf-8")
    with OpenAlexCatalog(workspace / "catalog.sqlite3") as catalog:
        pipeline = OpenAlexPipeline(snapshot=snapshot, catalog=catalog, workspace=workspace)
        result = await pipeline.add({"W2"})
        pipeline.mark_manual({"W2"})
        assert result.explicit_selected == 1
        resolver = FullTextResolver(output_dir=workspace / "fulltext", local_dir=local)
        try:
            materialized = await pipeline.materialize(
                resolver=resolver, content_mode="fulltext"
            )
        finally:
            await resolver.aclose()

        assert materialized.written == 1
        payload = json.loads(
            (workspace / "documents" / "W2.json").read_text(encoding="utf-8")
        )
        assert payload["content"] == "Complete article body."
        assert payload["abstract"] == "galaxy formation"
        assert payload["sources"][0]["display_name"] == "Journal of Voice"
