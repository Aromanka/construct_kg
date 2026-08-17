from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path


UTILS_DIR = Path(__file__).parents[1] / "src" / "utils"


def _load_module():
    sys.path.insert(0, str(UTILS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "sqlite_gold_to_neo4j", UTILS_DIR / "sqlite_gold_to_neo4j.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_gold_inspection_ignores_non_gold_tables(tmp_path, capsys) -> None:
    module = _load_module()
    database = tmp_path / "medical.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE entities (entity_id TEXT PRIMARY KEY);
            CREATE TABLE relation_types (relation_id TEXT PRIMARY KEY);
            CREATE TABLE assertions (assertion_id TEXT PRIMARY KEY);
            CREATE TABLE assertion_evidence (assertion_evidence_id TEXT PRIMARY KEY);
            CREATE TABLE raw_assertions (raw_assertion_id TEXT PRIMARY KEY);
            INSERT INTO entities VALUES ('entity-1');
            INSERT INTO relation_types VALUES ('relation-1');
            INSERT INTO assertions VALUES ('assertion-1');
            INSERT INTO raw_assertions VALUES ('raw-1');
            """
        )

    assert module.main(["inspect", "--sqlite", str(database)]) == 0
    output = capsys.readouterr().out
    assert "Gold 节点 entities: 1" in output
    assert "Gold 节点 assertions: 1" in output
    assert "raw_assertions" not in output


def test_gold_selection_requires_canonical_tables() -> None:
    module = _load_module()
    try:
        module.select_gold_tables({})
    except SystemExit as exc:
        assert "请先运行 canonicalize" in str(exc)
    else:
        raise AssertionError("missing Gold tables should be rejected")
