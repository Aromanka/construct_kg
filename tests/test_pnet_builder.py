from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from pnet.build_pnet import BuildConfig, BuildError, build_pnet


def _gold_fixture(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE entities (
                entity_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                description TEXT
            );
            CREATE TABLE entity_aliases (
                alias_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                alias TEXT NOT NULL
            );
            CREATE TABLE entity_external_ids (
                mapping_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                normalized_id TEXT NOT NULL,
                is_primary INTEGER NOT NULL
            );
            CREATE TABLE relation_types (
                relation_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL
            );
            CREATE TABLE raw_assertions (
                raw_assertion_id TEXT PRIMARY KEY,
                llm_confidence REAL NOT NULL
            );
            CREATE TABLE assertions (
                assertion_id TEXT PRIMARY KEY,
                raw_assertion_id TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL,
                object_entity_id TEXT NOT NULL,
                canonical_relation_id TEXT NOT NULL
            );

            INSERT INTO entities VALUES
                ('A', 'Heart sound', 'BIOMARKER', 'cardiac acoustic signal'),
                ('B', 'Intermediate B', 'PHENOTYPE', NULL),
                ('C', 'Source frontier C', 'DISEASE', NULL),
                ('X', 'Auscultation finding', 'BIOMARKER', NULL),
                ('T', 'Diabetes mellitus', 'DISEASE', NULL),
                ('U', 'Intermediate U', 'PHENOTYPE', NULL),
                ('W', 'Target frontier W', 'GENE', NULL);
            INSERT INTO entity_aliases VALUES ('alias-x', 'X', 'cardiac auscultation');
            INSERT INTO entity_external_ids VALUES ('ext-t', 'T', 'EFO:0000400', 1);
            INSERT INTO relation_types VALUES
                ('type-1', 'associated_with'),
                ('type-2', 'related_to');
            INSERT INTO raw_assertions VALUES
                ('raw-1', 0.8), ('raw-1b', 0.9), ('raw-2', 0.7),
                ('raw-3', 0.95), ('raw-4', 0.85), ('raw-5', 0.6), ('raw-6', 0.5);
            INSERT INTO assertions VALUES
                ('r1', 'raw-1', 'A', 'B', 'type-1'),
                ('r1b', 'raw-1b', 'A', 'B', 'type-2'),
                ('r2', 'raw-2', 'C', 'B', 'type-1'),
                ('r3', 'raw-3', 'U', 'T', 'type-1'),
                ('r4', 'raw-4', 'W', 'U', 'type-1'),
                ('r5', 'raw-5', 'A', 'X', 'type-2'),
                ('r6', 'raw-6', 'B', 'B', 'type-2');
            """
        )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_builds_complete_deterministic_pnet_with_carry_and_merged_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "gold.sqlite3"
    _gold_fixture(database)
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = dict(
        sqlite=database,
        text_s=("heart sound", "cardiac auscultation"),
        text_t=("diabetes",),
        source_max_hops=2,
        target_max_hops=2,
        traversal_mode="undirected",
        max_bridge_edges=10,
    )

    result = build_pnet(BuildConfig(output_dir=first, **common))
    build_pnet(BuildConfig(output_dir=second, **common))

    assert result.source_match_count == 2
    assert result.target_match_count == 1
    assert result.node_count == 9
    assert result.edge_count == 8
    assert result.bridge_edge_count == 2
    assert result.manifest["status"]["validation_passed"] is True
    assert result.manifest["graph"]["layer_widths"] == [2, 2, 2, 1, 1, 1]

    edges = _rows(first / "edges.tsv")
    merged = next(
        edge
        for edge in edges
        if edge["source_node_id"] == "s::d000::A"
        and edge["target_node_id"] == "s::d001::B"
    )
    assert merged["evidence_relation_ids"] == "r1|r1b"
    assert merged["relation_type"] == "associated_with|related_to"
    assert merged["confidence"] == "0.9"
    assert sum(edge["edge_kind"] == "structural_bridge" for edge in edges) == 2
    assert sum(edge["edge_kind"] == "structural_carry" for edge in edges) == 2

    nodes = _rows(first / "nodes.tsv")
    target_seed = next(node for node in nodes if node["node_id"] == "t::d000::T")
    assert target_seed["external_id"] == "EFO:0000400"
    assert all(node["display_name"] for node in nodes)

    rejected = _rows(first / "rejected_edges.tsv")
    assert any(row["rejection_reason"] == "same_layer" for row in rejected)
    assert any(row["rejection_reason"] == "self_loop" for row in rejected)

    for filename in ("graph.yaml", "nodes.tsv", "edges.tsv", "entity_matches.tsv"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_fails_instead_of_sampling_frontier_bridge(tmp_path: Path) -> None:
    database = tmp_path / "gold.sqlite3"
    _gold_fixture(database)
    config = BuildConfig(
        sqlite=database,
        output_dir=tmp_path / "output",
        text_s=("heart sound", "cardiac auscultation"),
        text_t=("diabetes",),
        source_max_hops=2,
        target_max_hops=2,
        max_bridge_edges=1,
    )

    with pytest.raises(BuildError, match="未抽样"):
        build_pnet(config)
    assert not config.output_dir.exists()


def test_directed_mode_uses_source_outbound_and_target_inbound(tmp_path: Path) -> None:
    database = tmp_path / "gold.sqlite3"
    _gold_fixture(database)
    output = tmp_path / "output"

    result = build_pnet(
        BuildConfig(
            sqlite=database,
            output_dir=output,
            text_s=("heart sound", "cardiac auscultation"),
            text_t=("diabetes",),
            source_max_hops=1,
            target_max_hops=1,
            traversal_mode="directed",
        )
    )

    assert result.node_count == 6
    assert result.edge_count == 5
    assert result.bridge_edge_count == 2
    node_ids = {node["node_id"] for node in _rows(output / "nodes.tsv")}
    assert "s::d001::B" in node_ids
    assert "s::d001::C" not in node_ids
    assert "t::d001::U" in node_ids


def test_fails_when_a_keyword_side_matches_nothing(tmp_path: Path) -> None:
    database = tmp_path / "gold.sqlite3"
    _gold_fixture(database)
    config = BuildConfig(
        sqlite=database,
        output_dir=tmp_path / "output",
        text_s=("missing source",),
        text_t=("diabetes",),
    )

    with pytest.raises(BuildError, match="起点关键词未匹配"):
        build_pnet(config)
    assert not config.output_dir.exists()
