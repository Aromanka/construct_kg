#!/usr/bin/env python3
"""Import a SQLite database into Neo4j and browse it in a small web UI.

The importer deliberately keeps two views of the data:

1. Every SQLite row becomes a ``SQLRow`` node, every column is retained as a
   property, and every declared foreign key becomes a ``SQL_FOREIGN_KEY`` edge.
2. The medical KG tables are projected into convenient graph edges:
   ``RAW_ASSERTION`` joins entity mentions, while ``ASSERTION`` joins canonical
   entities.  The source assertion row is still retained in the first view.

Only the Neo4j Python driver is required.  The web page has no JavaScript or
CSS dependency and is served by Python's standard library.
"""

# The embedded, dependency-free HTML/CSS/JavaScript is intentionally kept in this file.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sqlite3
import sys
import webbrowser
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_SQLITE = Path("data/medical_kg.sqlite3")
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_BATCH_SIZE = 500
MAX_WEB_ITEMS = 5_000
IDENTIFIER = re.compile(r"[^0-9A-Za-z_]")


def _quote_sql(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _cypher_label(name: str) -> str:
    safe = IDENTIFIER.sub("_", name)
    if not safe or safe[0].isdigit():
        safe = "T_" + safe
    return "SQL_" + safe


def _json_default(value: Any) -> str:
    if isinstance(value, bytes):
        return "base64:" + base64.b64encode(value).decode("ascii")
    return str(value)


def _neo4j_value(value: Any) -> Any:
    """Convert a SQLite value to a Neo4j property without discarding detail."""
    if isinstance(value, bytes):
        return "base64:" + base64.b64encode(value).decode("ascii")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _stable_key(table: str, columns: Sequence[str], values: Sequence[Any]) -> str:
    encoded = json.dumps(
        list(zip(columns, values, strict=True)),
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )
    return f"{table}:{encoded}"


def _chunks(items: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    has_rowid: bool

    @property
    def key_columns(self) -> tuple[str, ...]:
        return self.primary_key or ("__sqlite_rowid__",)


@dataclass(frozen=True)
class ForeignKeyInfo:
    child_table: str
    parent_table: str
    fk_id: int
    child_columns: tuple[str, ...]
    parent_columns: tuple[str, ...]


class SQLiteReader:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"SQLite database does not exist: {path}")
        uri = path.resolve().as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row
        self.tables = self._read_tables()

    def close(self) -> None:
        self.connection.close()

    def _read_tables(self) -> dict[str, TableInfo]:
        rows = self.connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables: dict[str, TableInfo] = {}
        for row in rows:
            name = str(row["name"])
            columns = self.connection.execute(
                f"PRAGMA table_info({_quote_sql(name)})"
            ).fetchall()
            ordered_pk = sorted(
                ((int(column["pk"]), str(column["name"])) for column in columns if column["pk"]),
                key=lambda item: item[0],
            )
            create_sql = str(row["sql"] or "").upper()
            tables[name] = TableInfo(
                name=name,
                columns=tuple(str(column["name"]) for column in columns),
                primary_key=tuple(name for _, name in ordered_pk),
                has_rowid="WITHOUT ROWID" not in create_sql,
            )
        return tables

    def count(self, table: str) -> int:
        return int(
            self.connection.execute(f"SELECT COUNT(*) FROM {_quote_sql(table)}").fetchone()[0]
        )

    def rows(self, table: TableInfo) -> Iterator[dict[str, Any]]:
        rowid = "rowid AS __sqlite_rowid__, " if not table.primary_key and table.has_rowid else ""
        cursor = self.connection.execute(
            f"SELECT {rowid}* FROM {_quote_sql(table.name)}"
        )
        for raw in cursor:
            row = dict(raw)
            key_values = [row[column] for column in table.key_columns]
            null_columns = [name for name in table.columns if row.get(name) is None]
            properties = {
                name: _neo4j_value(value)
                for name, value in row.items()
                if name != "__sqlite_rowid__" and value is not None
            }
            properties.update(
                {
                    "sql_key": _stable_key(table.name, table.key_columns, key_values),
                    "sql_table": table.name,
                    "sql_null_columns": json.dumps(null_columns, ensure_ascii=False),
                }
            )
            yield {"sql_key": properties["sql_key"], "properties": properties}

    def foreign_keys(self) -> list[ForeignKeyInfo]:
        result: list[ForeignKeyInfo] = []
        for child in self.tables.values():
            rows = self.connection.execute(
                f"PRAGMA foreign_key_list({_quote_sql(child.name)})"
            ).fetchall()
            groups: dict[int, list[sqlite3.Row]] = {}
            for row in rows:
                groups.setdefault(int(row["id"]), []).append(row)
            for fk_id, group in groups.items():
                group.sort(key=lambda row: int(row["seq"]))
                parent = str(group[0]["table"])
                if parent not in self.tables:
                    continue
                child_columns = tuple(str(row["from"]) for row in group)
                raw_parent_columns = tuple(str(row["to"] or "") for row in group)
                parent_columns = raw_parent_columns or self.tables[parent].primary_key
                if not all(parent_columns):
                    parent_columns = self.tables[parent].primary_key
                result.append(
                    ForeignKeyInfo(
                        child_table=child.name,
                        parent_table=parent,
                        fk_id=fk_id,
                        child_columns=child_columns,
                        parent_columns=parent_columns,
                    )
                )
        return result

    def foreign_key_rows(self, fk: ForeignKeyInfo) -> Iterator[dict[str, Any]]:
        child = self.tables[fk.child_table]
        parent = self.tables[fk.parent_table]
        child_key_select = _key_select("c", child.key_columns, "child")
        parent_key_select = _key_select("p", parent.key_columns, "parent")
        joins = " AND ".join(
            f"c.{_quote_sql(source)} = p.{_quote_sql(target)}"
            for source, target in zip(fk.child_columns, fk.parent_columns, strict=True)
        )
        query = (
            f"SELECT {child_key_select}, {parent_key_select} "
            f"FROM {_quote_sql(child.name)} AS c "
            f"JOIN {_quote_sql(parent.name)} AS p ON {joins}"
        )
        for row in self.connection.execute(query):
            child_values = [row[f"child_{index}"] for index in range(len(child.key_columns))]
            parent_values = [row[f"parent_{index}"] for index in range(len(parent.key_columns))]
            yield {
                "child_key": _stable_key(child.name, child.key_columns, child_values),
                "parent_key": _stable_key(parent.name, parent.key_columns, parent_values),
                "edge_key": f"{child.name}:{fk.fk_id}",
                "properties": {
                    "child_table": child.name,
                    "parent_table": parent.name,
                    "from_columns": json.dumps(fk.child_columns, ensure_ascii=False),
                    "to_columns": json.dumps(fk.parent_columns, ensure_ascii=False),
                },
            }


def _key_select(alias: str, columns: Sequence[str], prefix: str) -> str:
    parts = []
    for index, column in enumerate(columns):
        expression = f"{alias}.rowid" if column == "__sqlite_rowid__" else (
            f"{alias}.{_quote_sql(column)}"
        )
        parts.append(f"{expression} AS {_quote_sql(f'{prefix}_{index}')}")
    return ", ".join(parts)


class Neo4jImporter:
    def __init__(self, driver: Any, database: str | None, batch_size: int):
        self.driver = driver
        self.database = database
        self.batch_size = batch_size

    def _execute(self, query: str, **parameters: Any) -> Any:
        return self.driver.execute_query(query, database_=self.database, **parameters)

    def prepare(self, clear: bool) -> None:
        if clear:
            self._execute("MATCH (n:SQLRow) DETACH DELETE n")
        else:
            # Rebuild importer-owned edges so changed foreign keys/assertions do not leave stale
            # relationships. User relationships with other types remain untouched.
            self._execute(
                "MATCH (:SQLRow)-[r:SQL_FOREIGN_KEY|RAW_ASSERTION|ASSERTION]->(:SQLRow) "
                "DELETE r"
            )
        try:
            self._execute(
                "CREATE CONSTRAINT sql_row_key IF NOT EXISTS "
                "FOR (n:SQLRow) REQUIRE n.sql_key IS UNIQUE"
            )
        except Exception as exc:
            print(f"警告：未能创建唯一约束（导入仍可继续）：{exc}", file=sys.stderr)

    def import_table(self, table: TableInfo, rows: Iterable[dict[str, Any]]) -> int:
        label = _cypher_label(table.name)
        query = (
            "UNWIND $rows AS row "
            "MERGE (n:SQLRow {sql_key: row.sql_key}) "
            f"SET n:{label} SET n = row.properties"
        )
        count = 0
        for batch in _chunks(rows, self.batch_size):
            self._execute(query, rows=batch)
            count += len(batch)
        return count

    def import_foreign_keys(self, rows: Iterable[dict[str, Any]]) -> int:
        query = (
            "UNWIND $rows AS row "
            "MATCH (child:SQLRow {sql_key: row.child_key}) "
            "MATCH (parent:SQLRow {sql_key: row.parent_key}) "
            "MERGE (child)-[r:SQL_FOREIGN_KEY {edge_key: row.edge_key}]->(parent) "
            "SET r += row.properties"
        )
        count = 0
        for batch in _chunks(rows, self.batch_size):
            self._execute(query, rows=batch)
            count += len(batch)
        return count

    def project_knowledge_graph(self, tables: Mapping[str, TableInfo]) -> dict[str, int]:
        counts = {"raw_assertions": 0, "assertions": 0}
        if {"raw_assertions", "entity_mentions"}.issubset(tables):
            result = self._execute(
                "MATCH (a:SQL_raw_assertions), (s:SQL_entity_mentions), "
                "(o:SQL_entity_mentions) "
                "WHERE s.mention_id = a.subject_mention_id "
                "AND o.mention_id = a.object_mention_id "
                "MERGE (s)-[r:RAW_ASSERTION {sql_key: a.sql_key}]->(o) "
                "SET r += properties(a), r.relation = a.detailed_relation "
                "RETURN count(r) AS count"
            )
            counts["raw_assertions"] = _single_count(result)
        if {"assertions", "entities", "relation_types"}.issubset(tables):
            result = self._execute(
                "MATCH (a:SQL_assertions), (s:SQL_entities), (o:SQL_entities), "
                "(t:SQL_relation_types) "
                "WHERE s.entity_id = a.subject_entity_id "
                "AND o.entity_id = a.object_entity_id "
                "AND t.relation_id = a.canonical_relation_id "
                "OPTIONAL MATCH (e:SQL_assertion_evidence) "
                "WHERE e.assertion_id = a.assertion_id "
                "WITH a, s, o, t, count(e) AS support_count "
                "MERGE (s)-[r:ASSERTION {sql_key: a.sql_key}]->(o) "
                "SET r += properties(a), r.relation = t.canonical_name, "
                "r.support_count = support_count "
                "RETURN count(r) AS count"
            )
            counts["assertions"] = _single_count(result)
        return counts


def _single_count(result: Any) -> int:
    records = result.records if hasattr(result, "records") else result[0]
    return int(records[0]["count"]) if records else 0


def import_database(args: argparse.Namespace, driver: Any) -> None:
    reader = SQLiteReader(args.sqlite)
    importer = Neo4jImporter(driver, args.database, args.batch_size)
    try:
        importer.prepare(args.clear)
        print(f"SQLite: {args.sqlite.resolve()}")
        total_nodes = 0
        for table in reader.tables.values():
            imported = importer.import_table(table, reader.rows(table))
            total_nodes += imported
            print(f"  节点 {table.name}: {imported}")

        total_foreign_keys = 0
        for fk in reader.foreign_keys():
            imported = importer.import_foreign_keys(reader.foreign_key_rows(fk))
            total_foreign_keys += imported
            columns = ",".join(fk.child_columns)
            print(f"  外键 {fk.child_table}.{columns} -> {fk.parent_table}: {imported}")

        projected = importer.project_knowledge_graph(reader.tables)
        print(
            "导入完成："
            f"{total_nodes} 个 SQL 行节点，{total_foreign_keys} 条外键关系，"
            f"{projected['raw_assertions']} 条原始知识关系，"
            f"{projected['assertions']} 条规范化知识关系。"
        )
    finally:
        reader.close()


def inspect_database(path: Path) -> None:
    reader = SQLiteReader(path)
    try:
        print(f"SQLite: {path.resolve()}")
        for table in reader.tables.values():
            key = ", ".join(table.key_columns)
            print(f"  {table.name}: {reader.count(table.name)} 行；主键/行键: {key}")
        print(f"共 {len(reader.tables)} 张表，{len(reader.foreign_keys())} 个外键定义。")
    finally:
        reader.close()


def create_driver(
    uri: str, user: str, password: str | None, *, no_auth: bool = False
) -> Any:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise SystemExit(
            "缺少 Neo4j 驱动。请先运行：python -m pip install 'neo4j>=5,<7'"
        ) from exc
    auth = None if no_auth else (user, password)
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        driver.verify_connectivity()
    except Exception:
        driver.close()
        raise
    return driver


def _record_value(record: Any, key: str) -> Any:
    return record[key]


def query_graph(driver: Any, database: str | None, mode: str, limit: int) -> dict[str, Any]:
    if mode == "knowledge":
        mode = "canonical"
    filters = {
        "canonical": "WHERE type(r) = 'ASSERTION' ",
        "raw": "WHERE type(r) = 'RAW_ASSERTION' ",
        "all": "",
    }
    relationship_filter = filters.get(mode, filters["canonical"])
    query = (
        "MATCH (source:SQLRow)-[r]->(target:SQLRow) "
        f"{relationship_filter}"
        "RETURN source, labels(source) AS source_labels, r, type(r) AS relationship_type, "
        "target, labels(target) AS target_labels LIMIT $limit"
    )
    result = driver.execute_query(query, limit=limit, database_=database)
    records = result.records if hasattr(result, "records") else result[0]
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        source = _record_value(record, "source")
        target = _record_value(record, "target")
        source_properties = dict(source)
        target_properties = dict(target)
        source_key = str(source_properties["sql_key"])
        target_key = str(target_properties["sql_key"])
        nodes[source_key] = _web_node(
            source_key, source_properties, _record_value(record, "source_labels")
        )
        nodes[target_key] = _web_node(
            target_key, target_properties, _record_value(record, "target_labels")
        )
        relationship = _record_value(record, "r")
        properties = dict(relationship)
        relationship_type = str(_record_value(record, "relationship_type"))

        # Store assertion_id for later evidence lookup
        assertion_id = properties.get("assertion_id")

        edges.append(
            {
                "id": f"edge-{index}-{source_key}-{target_key}",
                "source": source_key,
                "target": target_key,
                "label": str(properties.get("relation") or relationship_type),
                "type": relationship_type,
                "properties": _json_safe(properties),
                "assertion_id": str(assertion_id) if assertion_id else None,
            }
        )

    # A relational database can legitimately contain lookup or newly-created rows without any
    # relationships. Include them as isolated nodes in the complete view instead of returning an
    # apparently empty database.
    if mode == "all" and len(nodes) < limit:
        remaining = limit - len(nodes)
        node_result = driver.execute_query(
            "MATCH (n:SQLRow) WHERE NOT n.sql_key IN $existing "
            "RETURN n, labels(n) AS node_labels LIMIT $limit",
            existing=list(nodes),
            limit=remaining,
            database_=database,
        )
        node_records = (
            node_result.records if hasattr(node_result, "records") else node_result[0]
        )
        for record in node_records:
            node = _record_value(record, "n")
            properties = dict(node)
            key = str(properties["sql_key"])
            nodes[key] = _web_node(
                key, properties, _record_value(record, "node_labels")
            )
    return {"nodes": list(nodes.values()), "edges": edges, "mode": mode}


def _web_node(key: str, properties: dict[str, Any], labels: Sequence[str]) -> dict[str, Any]:
    label = next(
        (
            str(properties[name])
            for name in ("canonical_name", "mention_text", "title", "canonical_relation", "name")
            if properties.get(name)
        ),
        str(properties.get("sql_table", key)),
    )
    kind = str(properties.get("entity_type") or properties.get("sql_table") or "SQLRow")
    return {
        "id": key,
        "label": label,
        "kind": kind,
        "labels": list(labels),
        "properties": _json_safe(properties),
    }


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))


def query_edge_evidence(sqlite_path: Path | None, assertion_id: str) -> dict[str, Any]:
    """Query SQLite for evidence details of a Gold assertion."""
    if not sqlite_path or not sqlite_path.is_file():
        return {"error": "SQLite database not available", "evidence": []}

    uri = sqlite_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

    try:
        # Join assertion_evidence → documents → raw_assertions
        query = """
            SELECT
                ae.evidence_text,
                ae.llm_confidence,
                d.document_id,
                d.title,
                d.file_path,
                d.doi,
                d.pmid,
                ra.detailed_relation
            FROM assertion_evidence ae
            JOIN documents d ON ae.document_id = d.document_id
            LEFT JOIN raw_assertions ra ON ae.raw_assertion_id = ra.raw_assertion_id
            WHERE ae.assertion_id = ?
            ORDER BY ae.llm_confidence DESC
        """
        rows = conn.execute(query, (assertion_id,)).fetchall()

        evidence = []
        for row in rows:
            evidence.append({
                "document_title": row["title"],
                "document_id": row["document_id"],
                "file_path": row["file_path"],
                "doi": row["doi"],
                "pmid": row["pmid"],
                "evidence_text": row["evidence_text"],
                "detailed_relation": row["detailed_relation"],
                "confidence": row["llm_confidence"],
            })

        return {"evidence": evidence}
    finally:
        conn.close()


class GraphRequestHandler(BaseHTTPRequestHandler):
    driver: Any = None
    database: str | None = None
    sqlite_path: Path | None = None

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, WEB_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/graph":
            parameters = parse_qs(parsed.query)
            mode = parameters.get("mode", ["canonical"])[0]
            if mode not in {"canonical", "raw", "all"}:
                mode = "canonical"
            try:
                requested = int(parameters.get("limit", ["500"])[0])
                limit = max(1, min(requested, MAX_WEB_ITEMS))
                payload = query_graph(self.driver, self.database, mode, limit)
                self._json(HTTPStatus.OK, payload)
            except Exception as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        if parsed.path == "/api/edge-evidence":
            parameters = parse_qs(parsed.query)
            assertion_id = parameters.get("assertion_id", [None])[0]
            if not assertion_id:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "assertion_id required"})
                return
            try:
                evidence = query_edge_evidence(self.sqlite_path, assertion_id)
                self._json(HTTPStatus.OK, evidence)
            except Exception as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, message: str, *args: Any) -> None:
        print(f"Web: {message % args}")

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(driver: Any, database: str | None, host: str, port: int, open_browser: bool, sqlite_path: Path | None = None) -> None:
    handler = type(
        "ConfiguredGraphHandler",
        (GraphRequestHandler,),
        {"driver": driver, "database": database, "sqlite_path": sqlite_path},
    )
    server = ThreadingHTTPServer((host, port), handler)
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{url_host}:{port}"
    print(f"知识图谱 Web 页面：{url} （Ctrl+C 停止）")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb 服务已停止。")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="完整导入 SQLite 到 Neo4j，并提供轻量级知识图谱 Web 页面。"
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("all", "import", "serve", "inspect"),
        default="all",
        help="all=导入后启动网页（默认）；import=仅导入；serve=仅网页；inspect=仅检查 SQLite",
    )
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE, help="SQLite 文件路径")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", DEFAULT_URI), help="Neo4j Bolt URI")
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", DEFAULT_USER), help="Neo4j 用户名")
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j 密码；推荐设置 NEO4J_PASSWORD 环境变量",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="连接已禁用身份验证的本地 Neo4j，不发送用户名和密码",
    )
    parser.add_argument(
        "--database", default=os.getenv("NEO4J_DATABASE"), help="Neo4j 数据库名（默认数据库可省略）"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批量写入行数")
    parser.add_argument(
        "--clear", action="store_true", help="导入前删除此前由本脚本导入的所有 SQLRow 节点"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Web 监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web 监听端口")
    parser.add_argument("--open-browser", action="store_true", help="启动 Web 服务后打开浏览器")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "inspect":
        inspect_database(args.sqlite)
        return 0
    if args.no_auth and args.password:
        parser.error("--no-auth 不能与 --password/NEO4J_PASSWORD 同时使用")
    if not args.no_auth and not args.password:
        parser.error("需要 --password/NEO4J_PASSWORD，或显式指定 --no-auth")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")

    driver = create_driver(args.uri, args.user, args.password, no_auth=args.no_auth)
    try:
        if args.action in {"all", "import"}:
            import_database(args, driver)
        if args.action in {"all", "serve"}:
            serve(driver, args.database, args.host, args.port, args.open_browser, args.sqlite)
    finally:
        driver.close()
    return 0


WEB_PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Medical KG Explorer</title>
<style>
:root{color-scheme:dark;--bg:#08111f;--panel:#101c2d;--line:#263b53;--text:#e8f0fa;--muted:#94a8bd;--accent:#55d6be}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif;overflow:hidden}
header{height:58px;display:flex;gap:12px;align-items:center;padding:9px 16px;background:#0d1828;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 12px 0 0;white-space:nowrap}input,select,button{color:var(--text);background:#13243a;border:1px solid #34506d;border-radius:6px;padding:7px 9px}
input[type=search]{width:min(360px,30vw)}button{cursor:pointer;background:#176b61}.status{color:var(--muted);margin-left:auto;white-space:nowrap}
main{display:grid;grid-template-columns:1fr 330px;height:calc(100vh - 58px)}canvas{width:100%;height:100%;cursor:grab}canvas:active{cursor:grabbing}
aside{background:var(--panel);border-left:1px solid var(--line);padding:16px;overflow:auto}aside h2{font-size:16px;margin:0 0 8px}.hint{color:var(--muted)}
.edge-highlight{background:#1a2d42;padding:4px 8px;border-radius:4px;margin:8px 0;border-left:3px solid var(--accent)}.flag{color:#ff9500;font-weight:600}
.tag{display:inline-block;padding:2px 7px;margin:2px;border-radius:10px;background:#20344b;color:#bcd0e2;font-size:12px}
table{width:100%;border-collapse:collapse;word-break:break-word}th,td{padding:6px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}th{color:var(--muted);width:34%}
@media(max-width:760px){main{grid-template-columns:1fr}aside{position:absolute;right:0;bottom:0;width:100%;height:38%;border-top:1px solid var(--line)}header{overflow-x:auto}.status{display:none}}
</style></head>
<body><header><h1>Medical KG Explorer</h1>
<select id="mode"><option value="canonical">Gold 规范化图</option><option value="raw">Bronze 原始图（诊断）</option><option value="all">完整数据库</option></select>
<input id="limit" type="number" min="1" max="5000" value="500" title="最多关系数">
<input id="search" type="search" placeholder="搜索名称、类型或任意属性…">
<button id="reload">加载</button><span class="status" id="status">准备加载</span></header>
<main><canvas id="graph"></canvas><aside id="detail"><h2>节点详情</h2><p class="hint">单击节点或关系查看 SQL 中保留的全部属性。拖拽节点，滚轮缩放，拖拽空白处平移。</p></aside></main>
<script>
const canvas=document.querySelector('#graph'),ctx=canvas.getContext('2d'),detail=document.querySelector('#detail'),statusEl=document.querySelector('#status');
let nodes=[],edges=[],scale=1,pan={x:0,y:0},drag=null,hover=null,frame=0,ticksLeft=0,animationPending=false;
const palette=['#55d6be','#68a8ff','#ffb45b','#d486ff','#ff718d','#8bd450','#ffd166','#5ad5e8'];
function hash(s){let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))|0;return Math.abs(h)}
function color(kind){return palette[hash(kind)%palette.length]}
function resize(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function escapeHtml(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function display(item,title){const props=item.properties||{};const isEdge=!!item.type;let html=`<h2>${escapeHtml(title)}</h2><div>${(item.labels||[item.type,item.kind]).filter(Boolean).map(x=>`<span class="tag">${escapeHtml(x)}</span>`).join('')}</div>`;if(isEdge&&props.negated)html+=`<div class="edge-highlight"><span class="flag">⚠ 否定关系</span></div>`;if(isEdge&&props.speculative)html+=`<div class="edge-highlight"><span class="flag">⚠ 推测性关系</span></div>`;if(isEdge&&props.support_count)html+=`<div class="edge-highlight">支持证据数：<strong>${props.support_count}</strong></div>`;let rows=Object.entries(props).sort(([a],[b])=>a.localeCompare(b)).map(([k,v])=>`<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(typeof v==='object'?JSON.stringify(v,null,2):v)}</td></tr>`).join('');detail.innerHTML=html+`<table>${rows}</table>`;if(isEdge&&item.assertion_id){detail.innerHTML+='<div id="evidence-loading" style="margin-top:16px;color:var(--muted)">加载证据详情...</div>';fetch(`/api/edge-evidence?assertion_id=${encodeURIComponent(item.assertion_id)}`).then(r=>r.json()).then(d=>{const container=document.querySelector('#evidence-loading');if(!container)return;if(d.error){container.innerHTML=`<p style="color:#f97316">加载证据失败: ${escapeHtml(d.error)}</p>`;return}if(!d.evidence||!d.evidence.length){container.innerHTML='<p style="color:var(--muted)">无证据详情。</p>';return}let evHtml='<h3 style="margin:16px 0 8px;font-size:15px">证据详情</h3>';d.evidence.forEach((ev,i)=>{evHtml+=`<div style="margin:12px 0;padding:10px;background:rgba(255,255,255,0.03);border-left:2px solid var(--accent);border-radius:4px"><p><strong>来源 ${i+1}:</strong> ${escapeHtml(ev.document_title||ev.document_id)}</p>`;if(ev.file_path)evHtml+=`<p style="font-size:0.9em;color:var(--muted)">文件: ${escapeHtml(ev.file_path)}</p>`;if(ev.doi)evHtml+=`<p style="font-size:0.9em">DOI: ${escapeHtml(ev.doi)}</p>`;if(ev.pmid)evHtml+=`<p style="font-size:0.9em">PMID: ${escapeHtml(ev.pmid)}</p>`;if(ev.detailed_relation)evHtml+=`<p><strong>原始关系短语:</strong> <em>"${escapeHtml(ev.detailed_relation)}"</em></p>`;if(ev.evidence_text)evHtml+=`<p><strong>证据文本:</strong> "${escapeHtml(ev.evidence_text)}"</p>`;if(ev.confidence!=null)evHtml+=`<p style="font-size:0.9em">置信度: ${ev.confidence}</p>`;evHtml+='</div>'});container.innerHTML=evHtml}).catch(err=>{const container=document.querySelector('#evidence-loading');if(container)container.innerHTML=`<p style="color:#f97316">加载失败: ${escapeHtml(err.message)}</p>`})}}
function resetPositions(){const w=canvas.clientWidth,h=canvas.clientHeight,n=Math.max(nodes.length,1),radius=Math.max(40,Math.min(w,h)*.42);nodes.forEach((x,i)=>{const a=i*Math.PI*(3-Math.sqrt(5)),r=radius*Math.sqrt((i+1)/n);x.x=w/2+Math.cos(a)*r;x.y=h/2+Math.sin(a)*r;x.vx=x.vy=0});pan={x:0,y:0};scale=1;restartSimulation(n>2000?80:n>1000?140:240)}
function simulate(){if(!nodes.length)return;const w=canvas.clientWidth,h=canvas.clientHeight,cx=w/2,cy=h/2,byId=new Map(nodes.map(n=>[n.id,n]));for(const e of edges){const a=byId.get(e.source),b=byId.get(e.target);if(!a||!b)continue;const dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),strength=Math.max(-1.5,Math.min(1.5,(d-105)*.008)),fx=dx/d*strength,fy=dy/d*strength;a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy}const neighbours=nodes.length>1000?24:60;for(let i=0;i<nodes.length;i++){for(let j=i+1;j<Math.min(nodes.length,i+neighbours);j++){const a=nodes[i],b=nodes[j],dx=b.x-a.x,dy=b.y-a.y,d2=dx*dx+dy*dy+25,f=Math.min(.035,18/d2);a.vx-=dx*f;a.vy-=dy*f;b.vx+=dx*f;b.vy+=dy*f}}for(const n of nodes){if(n===drag)continue;n.vx=(n.vx+(cx-n.x)*.0004)*.86;n.vy=(n.vy+(cy-n.y)*.0004)*.86;const speed=Math.hypot(n.vx,n.vy);if(speed>8){n.vx=n.vx/speed*8;n.vy=n.vy/speed*8}n.x=Math.max(cx-w*.7,Math.min(cx+w*.7,n.x+n.vx));n.y=Math.max(cy-h*.7,Math.min(cy+h*.7,n.y+n.vy));if(!Number.isFinite(n.x)||!Number.isFinite(n.y)){n.x=cx+(Math.random()-.5)*40;n.y=cy+(Math.random()-.5)*40;n.vx=n.vy=0}}}
function draw(){const w=canvas.clientWidth,h=canvas.clientHeight,showLabels=nodes.length<=1500;ctx.clearRect(0,0,w,h);ctx.save();ctx.translate(pan.x,pan.y);ctx.scale(scale,scale);const byId=new Map(nodes.map(n=>[n.id,n]));ctx.lineWidth=1/scale;ctx.font=`${11/scale}px system-ui`;for(const e of edges){const a=byId.get(e.source),b=byId.get(e.target);if(!a||!b)continue;ctx.strokeStyle=e===hover?'#fff':'#36516c';ctx.lineWidth=e===hover?2.5/scale:1/scale;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();if(showLabels&&scale>.65){const x=(a.x+b.x)/2,y=(a.y+b.y)/2;ctx.fillStyle='#91a6ba';ctx.fillText(e.label.slice(0,28),x,y)}}for(const n of nodes){ctx.beginPath();ctx.arc(n.x,n.y,n===hover?9/scale:7/scale,0,Math.PI*2);ctx.fillStyle=color(n.kind);ctx.fill();if(showLabels&&scale>.55){ctx.fillStyle='#e8f0fa';ctx.fillText(n.label.slice(0,34),n.x+10/scale,n.y+4/scale)}}ctx.restore()}
function animate(){if(ticksLeft<=0){animationPending=false;draw();return}simulate();ticksLeft--;draw();frame=requestAnimationFrame(animate)}
function restartSimulation(ticks=60){ticksLeft=Math.max(ticksLeft,ticks);if(!animationPending){animationPending=true;frame=requestAnimationFrame(animate)}}
function point(ev){const r=canvas.getBoundingClientRect();return{x:(ev.clientX-r.left-pan.x)/scale,y:(ev.clientY-r.top-pan.y)/scale}}
function nearest(p){let best=null,dist=16/scale;for(const n of nodes){const d=Math.hypot(n.x-p.x,n.y-p.y);if(d<dist){best=n;dist=d}}return best}
function nearestEdge(p){const byId=new Map(nodes.map(n=>[n.id,n])),threshold=8/scale;let best=null,minDist=threshold;for(const e of edges){const a=byId.get(e.source),b=byId.get(e.target);if(!a||!b)continue;const dx=b.x-a.x,dy=b.y-a.y,len2=dx*dx+dy*dy;if(len2<1)continue;const t=Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/len2)),cx=a.x+t*dx,cy=a.y+t*dy,d=Math.hypot(p.x-cx,p.y-cy);if(d<minDist){best=e;minDist=d}}return best}
canvas.addEventListener('mousedown',e=>{const p=point(e);drag=nearest(p)||{pan:true,sx:e.clientX,sy:e.clientY,px:pan.x,py:pan.y}});
addEventListener('mousemove',e=>{if(!drag)return;if(drag.pan){pan.x=drag.px+e.clientX-drag.sx;pan.y=drag.py+e.clientY-drag.sy;draw()}else{const p=point(e);drag.x=p.x;drag.y=p.y;drag.vx=drag.vy=0;restartSimulation(30)}});
addEventListener('mouseup',()=>{if(drag&&!drag.pan)restartSimulation(80);drag=null});canvas.addEventListener('wheel',e=>{e.preventDefault();scale=Math.max(.15,Math.min(4,scale*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
canvas.addEventListener('click',e=>{const p=point(e),n=nearest(p),edge=nearestEdge(p);if(n)display(n,n.label);else if(edge)display(edge,edge.label||edge.type)});
document.querySelector('#search').addEventListener('input',e=>{const q=e.target.value.trim().toLowerCase();if(q){const node=nodes.find(n=>JSON.stringify(n).toLowerCase().includes(q)),edge=edges.find(e=>JSON.stringify(e).toLowerCase().includes(q));hover=node||edge||null;if(hover){display(hover,hover.label||hover.type);if(node){pan.x=canvas.clientWidth/2-hover.x*scale;pan.y=canvas.clientHeight/2-hover.y*scale}}}else{hover=null}draw()});
async function load(){statusEl.textContent='加载中…';try{const mode=document.querySelector('#mode').value,limit=document.querySelector('#limit').value;const response=await fetch(`/api/graph?mode=${mode}&limit=${limit}`),data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);nodes=data.nodes;edges=data.edges;resetPositions();const emptyHint=mode==='canonical'&&!edges.length?' · 尚无 Gold 关系，请先运行 canonicalize':'';statusEl.textContent=`${nodes.length} 节点 · ${edges.length} 关系${emptyHint}`;detail.innerHTML='<h2>图谱概览</h2><p class="hint">'+statusEl.textContent+'。单击节点或关系查看全部属性。</p>'}catch(e){statusEl.textContent='加载失败';detail.innerHTML=`<h2>错误</h2><p>${escapeHtml(e.message)}</p>`}}
document.querySelector('#reload').onclick=load;document.querySelector('#mode').onchange=load;addEventListener('resize',resize);resize();load();
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
