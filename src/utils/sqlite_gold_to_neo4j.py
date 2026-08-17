#!/usr/bin/env python3
"""Import only the canonical Gold knowledge graph from SQLite into Neo4j.

Unlike ``sqlite_to_neo4j.py``, this entry point deliberately skips Bronze rows,
documents, extraction jobs, and other operational tables. It imports only the
tables needed by the canonical graph and then serves the existing Web explorer.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from . import sqlite_to_neo4j as shared
else:
    import sqlite_to_neo4j as shared


GOLD_TABLES = ("entities", "relation_types", "assertions", "assertion_evidence")
REQUIRED_GOLD_TABLES = frozenset({"entities", "relation_types", "assertions"})
DEFAULT_DELETE_BATCH_SIZE = 5_000


def select_gold_tables(
    tables: Mapping[str, shared.TableInfo],
) -> dict[str, shared.TableInfo]:
    """Return Gold tables in dependency order and reject a non-canonicalized DB."""
    missing = sorted(REQUIRED_GOLD_TABLES.difference(tables))
    if missing:
        names = ", ".join(missing)
        raise SystemExit(f"SQLite 缺少 Gold 表：{names}；请先运行 canonicalize。")
    return {name: tables[name] for name in GOLD_TABLES if name in tables}


class GoldNeo4jImporter(shared.Neo4jImporter):
    """Small-memory importer whose owned nodes are deleted in bounded batches."""

    def _delete_nodes_in_batches(self, label: str, batch_size: int) -> int:
        total = 0
        while True:
            result = self._execute(
                f"MATCH (n:{label}) WITH n LIMIT $limit "
                "DETACH DELETE n RETURN count(*) AS count",
                limit=batch_size,
            )
            deleted = shared._single_count(result)
            total += deleted
            if deleted < batch_size:
                return total

    def prepare_gold(self, *, clear: bool, delete_batch_size: int) -> None:
        # --clear also removes nodes left by an interrupted legacy full import.
        target_label = "SQLRow" if clear else "GoldSQLRow"
        deleted = self._delete_nodes_in_batches(target_label, delete_batch_size)
        if deleted:
            print(f"已分批删除 {deleted} 个旧 {target_label} 节点。")

        # A legacy import may have canonical edges between untagged SQLRow nodes.
        self._execute("MATCH (:SQLRow)-[r:ASSERTION]->(:SQLRow) DELETE r")
        self._execute(
            "CREATE CONSTRAINT sql_row_key IF NOT EXISTS "
            "FOR (n:SQLRow) REQUIRE n.sql_key IS UNIQUE"
        )

    def import_gold_table(
        self, table: shared.TableInfo, rows: Iterable[dict[str, Any]]
    ) -> int:
        label = shared._cypher_label(table.name)
        query = (
            "UNWIND $rows AS row "
            "MERGE (n:SQLRow {sql_key: row.sql_key}) "
            f"SET n:GoldSQLRow SET n:{label} SET n = row.properties"
        )
        count = 0
        for batch in shared._chunks(rows, self.batch_size):
            self._execute(query, rows=batch)
            count += len(batch)
        return count


def inspect_gold_database(path: Path) -> None:
    reader = shared.SQLiteReader(path)
    try:
        tables = select_gold_tables(reader.tables)
        total = 0
        print(f"SQLite: {path.resolve()}")
        for table in tables.values():
            count = reader.count(table.name)
            total += count
            print(f"  Gold 节点 {table.name}: {count}")
        print(f"将导入 {len(tables)} 张 Gold 表、共 {total} 个 SQL 行节点。")
    finally:
        reader.close()


def gold_relationship_rows(reader: shared.SQLiteReader) -> Iterable[dict[str, Any]]:
    """Stream canonical edges, resolving their small lookup data in SQLite."""
    entity_table = reader.tables["entities"]
    assertion_table = reader.tables["assertions"]
    relation_names = {
        str(row["relation_id"]): str(row["canonical_name"])
        for row in reader.connection.execute(
            "SELECT relation_id, canonical_name FROM relation_types"
        )
    }
    evidence_counts: dict[str, int] = {}
    if "assertion_evidence" in reader.tables:
        evidence_counts = {
            str(row["assertion_id"]): int(row["support_count"])
            for row in reader.connection.execute(
                "SELECT assertion_id, COUNT(*) AS support_count "
                "FROM assertion_evidence GROUP BY assertion_id"
            )
        }

    for row in reader.rows(assertion_table):
        properties = dict(row["properties"])
        assertion_id = str(properties["assertion_id"])
        relation_id = str(properties["canonical_relation_id"])
        source_id = properties["subject_entity_id"]
        target_id = properties["object_entity_id"]
        properties["relation"] = relation_names[relation_id]
        properties["support_count"] = evidence_counts.get(assertion_id, 0)
        yield {
            "source_key": shared._stable_key(
                "entities", entity_table.key_columns, (source_id,)
            ),
            "target_key": shared._stable_key(
                "entities", entity_table.key_columns, (target_id,)
            ),
            "sql_key": properties["sql_key"],
            "properties": properties,
        }


def import_gold_relationships(
    importer: GoldNeo4jImporter, rows: Iterable[dict[str, Any]]
) -> int:
    query = (
        "UNWIND $rows AS row "
        "MATCH (source:SQLRow {sql_key: row.source_key}) "
        "MATCH (target:SQLRow {sql_key: row.target_key}) "
        "MERGE (source)-[r:ASSERTION {sql_key: row.sql_key}]->(target) "
        "SET r = row.properties"
    )
    count = 0
    for batch in shared._chunks(rows, importer.batch_size):
        importer._execute(query, rows=batch)
        count += len(batch)
    return count


def import_gold_database(args: argparse.Namespace, driver: Any) -> None:
    reader = shared.SQLiteReader(args.sqlite)
    importer = GoldNeo4jImporter(driver, args.database, args.batch_size)
    try:
        tables = select_gold_tables(reader.tables)
        importer.prepare_gold(clear=args.clear, delete_batch_size=args.delete_batch_size)
        print(f"SQLite: {args.sqlite.resolve()}")

        total_nodes = 0
        for table in tables.values():
            imported = importer.import_gold_table(table, reader.rows(table))
            total_nodes += imported
            print(f"  Gold 节点 {table.name}: {imported}")

        relationships = import_gold_relationships(importer, gold_relationship_rows(reader))
        print(
            "Gold 导入完成："
            f"{total_nodes} 个 SQL 行节点，"
            f"{relationships} 条规范化知识关系。"
        )
    finally:
        reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只导入 SQLite 的 Gold 规范化图谱，并提供轻量级 Web 页面。"
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("all", "import", "serve", "inspect"),
        default="all",
        help="all=导入后启动网页（默认）；import=仅导入；serve=仅网页；inspect=仅检查 Gold 表",
    )
    parser.add_argument(
        "--sqlite", type=Path, default=shared.DEFAULT_SQLITE, help="主 SQLite 文件路径"
    )
    parser.add_argument(
        "--uri", default=os.getenv("NEO4J_URI", shared.DEFAULT_URI), help="Neo4j Bolt URI"
    )
    parser.add_argument(
        "--user", default=os.getenv("NEO4J_USER", shared.DEFAULT_USER), help="Neo4j 用户名"
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j 密码；推荐设置 NEO4J_PASSWORD 环境变量",
    )
    parser.add_argument(
        "--no-auth", action="store_true", help="连接已禁用身份验证的本地 Neo4j"
    )
    parser.add_argument(
        "--database", default=os.getenv("NEO4J_DATABASE"), help="Neo4j 数据库名"
    )
    parser.add_argument(
        "--batch-size", type=int, default=shared.DEFAULT_BATCH_SIZE, help="批量写入行数"
    )
    parser.add_argument(
        "--delete-batch-size",
        type=int,
        default=DEFAULT_DELETE_BATCH_SIZE,
        help="--clear 分批删除的节点数",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="分批删除此前导入的全部 SQLRow 节点；不指定时只替换 GoldSQLRow",
    )
    parser.add_argument("--host", default=shared.DEFAULT_HOST, help="Web 监听地址")
    parser.add_argument("--port", type=int, default=shared.DEFAULT_PORT, help="Web 监听端口")
    parser.add_argument("--open-browser", action="store_true", help="启动后打开浏览器")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "inspect":
        inspect_gold_database(args.sqlite)
        return 0
    if args.no_auth and args.password:
        parser.error("--no-auth 不能与 --password/NEO4J_PASSWORD 同时使用")
    if not args.no_auth and not args.password:
        parser.error("需要 --password/NEO4J_PASSWORD，或显式指定 --no-auth")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if args.delete_batch_size < 1:
        parser.error("--delete-batch-size 必须大于 0")

    driver = shared.create_driver(args.uri, args.user, args.password, no_auth=args.no_auth)
    try:
        if args.action in {"all", "import"}:
            import_gold_database(args, driver)
        if args.action in {"all", "serve"}:
            shared.serve(driver, args.database, args.host, args.port, args.open_browser)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
