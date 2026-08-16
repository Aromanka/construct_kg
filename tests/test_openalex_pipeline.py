from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from medical_kg.openalex.catalog import OpenAlexCatalog
from medical_kg.openalex.cli import _selection_options, build_parser
from medical_kg.openalex.filtering import MEDICAL_BROAD_FIELDS, WorkFilter
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
    field_id: int | None = 27,
) -> dict[str, object]:
    positions = {token: [index] for index, token in enumerate(abstract.split())}
    record: dict[str, object] = {
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
    if field_id is not None:
        record["topics"] = [
            {
                "id": "https://openalex.org/T1",
                "display_name": "Example topic",
                "field": {"id": f"https://openalex.org/fields/{field_id}"},
            }
        ]
    return record


def _snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "openalex-snapshot"
    works = root / "data" / "works" / "updated_date=2025-01-01"
    sources = root / "data" / "sources" / "updated_date=2025-01-01"
    works.mkdir(parents=True)
    sources.mkdir(parents=True)
    records = [
        _work("W1", "Diabetes voice biomarkers", "voice changes in diabetes"),
        _work(
            "W2",
            "Unrelated astronomy",
            "galaxy formation",
            fulltext=False,
            field_id=17,
        ),
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


def test_cli_enables_medical_field_gate_and_treats_lone_id_as_explicit() -> None:
    parser = build_parser()
    unfiltered = parser.parse_args(["select", "snapshot"])
    assert set(_selection_options(unfiltered).work_filter.allowed_field_ids or ()) == set(
        MEDICAL_BROAD_FIELDS
    )

    disabled = parser.parse_args(
        ["select", "snapshot", "--no-medical-field-filter"]
    )
    with pytest.raises(ValueError, match="At least one filter"):
        _selection_options(disabled)

    explicit = parser.parse_args(["select", "snapshot", "--include-id", "W1"])
    assert _selection_options(explicit).max_candidates == 0


def test_default_field_gate_accepts_only_medical_and_biomedical_fields() -> None:
    assert MEDICAL_BROAD_FIELDS == frozenset({13, 24, 27, 28, 30, 36})
    medical = OpenAlexWork.from_raw(_work("W10", "Medical", "abstract", field_id=27))
    biomedical = OpenAlexWork.from_raw(
        _work("W11", "Biomedical", "abstract", field_id=30)
    )
    unrelated = OpenAlexWork.from_raw(
        _work("W12", "Astronomy", "abstract", field_id=17)
    )
    missing = OpenAlexWork.from_raw(_work("W13", "Unknown", "abstract", field_id=None))

    work_filter = WorkFilter()
    assert work_filter.match(medical)[0]
    assert work_filter.match(biomedical)[0]
    assert not work_filter.match(unrelated)[0]
    assert not work_filter.match(missing)[0]


def test_snapshot_strictly_discovers_gzip_parts(tmp_path: Path) -> None:
    snapshot = OpenAlexSnapshot(_snapshot(tmp_path))

    assert [path.name for path in snapshot.entity_files("works")] == ["part_0000.gz"]
    assert [work.work_id for work in snapshot.iter_works()] == ["W1", "W2", "W3"]


@pytest.mark.asyncio
async def test_select_applies_default_medical_field_gate(tmp_path: Path) -> None:
    snapshot = OpenAlexSnapshot(_snapshot(tmp_path))
    workspace = tmp_path / "feature"
    with OpenAlexCatalog(workspace / "catalog.sqlite3") as catalog:
        pipeline = OpenAlexPipeline(snapshot=snapshot, catalog=catalog, workspace=workspace)
        statistics = await pipeline.select(SelectionOptions())

        assert statistics.scanned == 3
        assert statistics.candidates == 2
        assert statistics.selected == 2
        assert catalog.get("W1") is not None
        assert catalog.get("W2") is None
        assert catalog.get("W3") is not None


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
