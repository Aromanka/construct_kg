#!/usr/bin/env python3
"""Import PNet TSV files into Neo4j and visualize with an interactive web UI.

This script reads the nodes.tsv and edges.tsv produced by build_pnet.py and
imports them into Neo4j as a layered directed acyclic graph. The web interface
provides an interactive visualization with layer-based layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import webbrowser
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_PNET_DIR = Path("src/pnet/pnet_output")
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_BATCH_SIZE = 500
DEFAULT_DELETE_BATCH_SIZE = 5_000
MAX_WEB_ITEMS = 5_000


def _parse_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def _chunks(items: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class PNetReader:
    def __init__(self, pnet_dir: Path):
        if not pnet_dir.is_dir():
            raise FileNotFoundError(f"PNet directory does not exist: {pnet_dir}")
        self.pnet_dir = pnet_dir
        self.nodes_file = pnet_dir / "nodes.tsv"
        self.edges_file = pnet_dir / "edges.tsv"
        if not self.nodes_file.is_file():
            raise FileNotFoundError(f"nodes.tsv not found: {self.nodes_file}")
        if not self.edges_file.is_file():
            raise FileNotFoundError(f"edges.tsv not found: {self.edges_file}")

    def read_nodes(self) -> Iterator[dict[str, Any]]:
        with self.nodes_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                properties = {
                    "node_id": row["node_id"],
                    "entity_id": row["entity_id"],
                    "layer": row["layer"],
                    "node_type": row["node_type"],
                    "display_name": row["display_name"],
                    "is_fallback": _parse_bool(row["is_fallback"]),
                    "source": row["source"],
                    "is_structural": _parse_bool(row["is_structural"]),
                }
                # Add optional fields if present
                for field in ["external_id", "original_node_id", "description", "side",
                              "matched_keywords", "source_distance", "target_distance",
                              "lateral_steps", "terminal_entity_id"]:
                    if field in row and row[field]:
                        if field in {"source_distance", "target_distance", "lateral_steps", "bfs_depth"}:
                            properties[field] = int(row[field]) if row[field] else 0
                        else:
                            properties[field] = row[field]
                if "bfs_depth" in row and row["bfs_depth"]:
                    properties["bfs_depth"] = int(row["bfs_depth"])
                yield {"node_id": row["node_id"], "properties": properties}

    def read_edges(self) -> Iterator[dict[str, Any]]:
        with self.edges_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                properties = {
                    "relation_type": row["relation_type"],
                    "knowledge_source": row["knowledge_source"],
                    "confidence": float(row["confidence"]) if row["confidence"] else 1.0,
                    "is_fallback": _parse_bool(row["is_fallback"]),
                    "enabled": _parse_bool(row.get("enabled", "true")),
                    "edge_kind": row.get("edge_kind", "kg_bfs"),
                    "is_structural": _parse_bool(row["is_structural"]),
                }
                # Add optional fields
                for field in ["evidence_relation_ids", "original_source_entity_id",
                              "original_target_entity_id", "original_direction",
                              "traversal_direction"]:
                    if field in row and row[field]:
                        properties[field] = row[field]
                yield {
                    "source_node_id": row["source_node_id"],
                    "target_node_id": row["target_node_id"],
                    "properties": properties,
                }

    def count_nodes(self) -> int:
        with self.nodes_file.open(encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # -1 for header

    def count_edges(self) -> int:
        with self.edges_file.open(encoding="utf-8") as f:
            return sum(1 for _ in f) - 1


class PNetNeo4jImporter:
    def __init__(self, driver: Any, database: str | None, batch_size: int):
        self.driver = driver
        self.database = database
        self.batch_size = batch_size

    def _execute(self, query: str, **parameters: Any) -> Any:
        return self.driver.execute_query(query, database_=self.database, **parameters)

    def _delete_nodes_in_batches(self, label: str, batch_size: int) -> int:
        total = 0
        while True:
            result = self._execute(
                f"MATCH (n:{label}) WITH n LIMIT $limit "
                "DETACH DELETE n RETURN count(*) AS count",
                limit=batch_size,
            )
            records = result.records if hasattr(result, "records") else result[0]
            deleted = int(records[0]["count"]) if records else 0
            total += deleted
            if deleted < batch_size:
                return total

    def prepare(self, *, clear: bool, delete_batch_size: int) -> None:
        target_label = "PNetNode" if clear else "PNetNode"
        if clear:
            deleted = self._delete_nodes_in_batches(target_label, delete_batch_size)
            if deleted:
                print(f"已分批删除 {deleted} 个旧 {target_label} 节点。")
        self._execute(
            "CREATE CONSTRAINT pnet_node_id IF NOT EXISTS "
            "FOR (n:PNetNode) REQUIRE n.node_id IS UNIQUE"
        )

    def import_nodes(self, nodes: Iterator[dict[str, Any]]) -> int:
        query = (
            "UNWIND $rows AS row "
            "MERGE (n:PNetNode {node_id: row.node_id}) "
            "SET n = row.properties"
        )
        count = 0
        batch = []
        for node in nodes:
            batch.append(node)
            if len(batch) >= self.batch_size:
                self._execute(query, rows=batch)
                count += len(batch)
                batch = []
        if batch:
            self._execute(query, rows=batch)
            count += len(batch)
        return count

    def import_edges(self, edges: Iterator[dict[str, Any]]) -> int:
        query = (
            "UNWIND $rows AS row "
            "MATCH (source:PNetNode {node_id: row.source_node_id}) "
            "MATCH (target:PNetNode {node_id: row.target_node_id}) "
            "MERGE (source)-[r:PNET_EDGE]->(target) "
            "SET r = row.properties"
        )
        count = 0
        batch = []
        for edge in edges:
            batch.append(edge)
            if len(batch) >= self.batch_size:
                self._execute(query, rows=batch)
                count += len(batch)
                batch = []
        if batch:
            self._execute(query, rows=batch)
            count += len(batch)
        return count


def inspect_pnet(pnet_dir: Path) -> None:
    reader = PNetReader(pnet_dir)
    node_count = reader.count_nodes()
    edge_count = reader.count_edges()
    print(f"PNet 目录: {pnet_dir.resolve()}")
    print(f"  节点数: {node_count}")
    print(f"  边数: {edge_count}")
    graph_yaml = pnet_dir / "graph.yaml"
    if graph_yaml.is_file():
        import yaml
        with graph_yaml.open(encoding="utf-8") as f:
            graph = yaml.safe_load(f)
        print(f"  算法: {graph.get('algorithm', {}).get('name', 'unknown')}")
        print(f"  层数: {len(graph.get('layers', []))}")


def import_pnet(args: argparse.Namespace, driver: Any) -> None:
    reader = PNetReader(args.pnet_dir)
    importer = PNetNeo4jImporter(driver, args.database, args.batch_size)
    importer.prepare(clear=args.clear, delete_batch_size=args.delete_batch_size)
    print(f"PNet 目录: {args.pnet_dir.resolve()}")

    node_count = importer.import_nodes(reader.read_nodes())
    print(f"  导入节点: {node_count}")

    edge_count = importer.import_edges(reader.read_edges())
    print(f"  导入边: {edge_count}")
    print(f"PNet 导入完成: {node_count} 个节点, {edge_count} 条边")


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


def query_pnet(driver: Any, database: str | None, limit: int) -> dict[str, Any]:
    query = (
        "MATCH (source:PNetNode)-[r:PNET_EDGE]->(target:PNetNode) "
        "RETURN source, r, target LIMIT $limit"
    )
    result = driver.execute_query(query, limit=limit, database_=database)
    records = result.records if hasattr(result, "records") else result[0]
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for record in records:
        source = dict(record["source"])
        target = dict(record["target"])
        relationship = dict(record["r"])

        source_id = source["node_id"]
        target_id = target["node_id"]

        if source_id not in nodes:
            nodes[source_id] = {
                "id": source_id,
                "label": source.get("display_name", source_id),
                "type": source.get("node_type", ""),
                "layer": source.get("layer", ""),
                "is_structural": source.get("is_structural", False),
                "properties": source,
            }
        if target_id not in nodes:
            nodes[target_id] = {
                "id": target_id,
                "label": target.get("display_name", target_id),
                "type": target.get("node_type", ""),
                "layer": target.get("layer", ""),
                "is_structural": target.get("is_structural", False),
                "properties": target,
            }

        edges.append({
            "source": source_id,
            "target": target_id,
            "relation": relationship.get("relation_type", ""),
            "edge_kind": relationship.get("edge_kind", ""),
            "is_structural": relationship.get("is_structural", False),
            "properties": relationship,
        })

    return {"nodes": list(nodes.values()), "edges": edges}


class PNetRequestHandler(BaseHTTPRequestHandler):
    driver: Any
    database: str | None

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(WEB_PAGE.encode("utf-8"))
        elif parsed.path == "/api/graph":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", [str(MAX_WEB_ITEMS)])[0])
            limit = min(limit, MAX_WEB_ITEMS)
            try:
                data = query_pnet(self.driver, self.database, limit)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            except Exception as exc:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error = {"error": str(exc)}
                self.wfile.write(json.dumps(error, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def serve(driver: Any, database: str | None, host: str, port: int, open_browser: bool) -> None:
    handler = type(
        "ConfiguredPNetHandler",
        (PNetRequestHandler,),
        {"driver": driver, "database": database},
    )
    server = ThreadingHTTPServer((host, port), handler)
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{url_host}:{port}"
    print(f"PNet Web 页面：{url} （Ctrl+C 停止）")
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
        description="导入 PNet TSV 文件到 Neo4j，并提供交互式 Web 可视化。"
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("all", "import", "serve", "inspect"),
        default="all",
        help="all=导入后启动网页（默认）；import=仅导入；serve=仅网页；inspect=仅检查 PNet",
    )
    parser.add_argument(
        "--pnet-dir",
        type=Path,
        default=DEFAULT_PNET_DIR,
        help="PNet 输出目录路径",
    )
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", DEFAULT_URI), help="Neo4j Bolt URI")
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", DEFAULT_USER), help="Neo4j 用户名")
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j 密码；推荐设置 NEO4J_PASSWORD 环境变量",
    )
    parser.add_argument("--no-auth", action="store_true", help="连接已禁用身份验证的本地 Neo4j")
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE"), help="Neo4j 数据库名")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批量写入行数")
    parser.add_argument(
        "--delete-batch-size",
        type=int,
        default=DEFAULT_DELETE_BATCH_SIZE,
        help="--clear 分批删除的节点数",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="分批删除此前导入的全部 PNetNode 节点",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Web 监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web 监听端口")
    parser.add_argument("--open-browser", action="store_true", help="启动后打开浏览器")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "inspect":
        inspect_pnet(args.pnet_dir)
        return 0
    if args.no_auth and args.password:
        parser.error("--no-auth 不能与 --password/NEO4J_PASSWORD 同时使用")
    if not args.no_auth and not args.password:
        parser.error("需要 --password/NEO4J_PASSWORD，或显式指定 --no-auth")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if args.delete_batch_size < 1:
        parser.error("--delete-batch-size 必须大于 0")

    driver = create_driver(args.uri, args.user, args.password, no_auth=args.no_auth)
    try:
        if args.action in {"all", "import"}:
            import_pnet(args, driver)
        if args.action in {"all", "serve"}:
            serve(driver, args.database, args.host, args.port, args.open_browser)
    finally:
        driver.close()
    return 0


WEB_PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PNet Explorer</title>
<style>
:root{color-scheme:dark;--bg:#0B0E14;--panel:#0F172A;--line:#1E293B;--text:#F8FAFC;--muted:#94A3B8;--accent:#38BDF8;--structural:#F97316}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Inter",-apple-system,sans-serif;overflow:hidden}
header{height:58px;display:flex;gap:12px;align-items:center;padding:9px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:17px;font-weight:600;margin:0 12px 0 0;white-space:nowrap;color:var(--accent)}
input,select,button{color:var(--text);background:#1E293B;border:1px solid #334155;border-radius:6px;padding:7px 11px;font-size:13px}
button{cursor:pointer;background:#0F766E;border-color:#0F766E;font-weight:500}button:hover{background:#115E59}
.status{color:var(--muted);margin-left:auto;white-space:nowrap;font-size:13px}
main{display:grid;grid-template-columns:1fr 340px;height:calc(100vh - 58px)}
canvas{width:100%;height:100%;cursor:grab;background:var(--bg)}canvas:active{cursor:grabbing}
aside{background:var(--panel);border-left:1px solid var(--line);padding:18px;overflow:auto}
aside h2{font-size:15px;font-weight:600;margin:0 0 10px;color:var(--accent)}.hint{color:var(--muted);font-size:13px}
.tag{display:inline-block;padding:3px 9px;margin:2px;border-radius:4px;background:#1E293B;color:#94A3B8;font-size:11px;border:1px solid #334155}
.tag.structural{background:#431407;color:#FED7AA;border-color:#7C2D12}
table{width:100%;border-collapse:collapse;word-break:break-word;margin-top:12px}
th,td{padding:7px 4px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left;font-size:13px}
th{color:var(--muted);width:35%;font-weight:500}td{font-family:"SF Mono",Consolas,monospace;font-size:12px}
@media(max-width:760px){main{grid-template-columns:1fr}aside{position:absolute;right:0;bottom:0;width:100%;height:38%;border-top:1px solid var(--line)}}
</style></head>
<body><header><h1>PNet Explorer</h1>
<button id="loadBtn">Load PNet</button>
<span class="status" id="status">Ready</span>
</header>
<main><canvas id="canvas"></canvas><aside>
<h2>PNet Information</h2>
<p class="hint">Click "Load PNet" to visualize the path network. Nodes are arranged by layer. Drag to pan.</p>
<div id="info"></div>
</aside></main>
<script>
const canvas=document.getElementById("canvas"),ctx=canvas.getContext("2d"),status=document.getElementById("status"),info=document.getElementById("info"),loadBtn=document.getElementById("loadBtn");
let nodes=[],edges=[],camera={x:0,y:0,scale:1},drag=null;

function resize(){canvas.width=canvas.offsetWidth*devicePixelRatio;canvas.height=canvas.offsetHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);draw()}
window.addEventListener("resize",resize);resize();

loadBtn.onclick=async()=>{
  status.textContent="Loading...";
  try{
    const r=await fetch("/api/graph?limit=5000");
    const data=await r.json();
    if(data.error){status.textContent="Error: "+data.error;return}
    nodes=data.nodes;edges=data.edges;
    layout();draw();
    info.innerHTML=`<p><strong>${nodes.length}</strong> nodes, <strong>${edges.length}</strong> edges</p>`;
    status.textContent="Loaded "+nodes.length+" nodes";
  }catch(e){status.textContent="Error: "+e.message}
};

function layout(){
  const layers={};
  nodes.forEach(n=>{if(!layers[n.layer])layers[n.layer]=[];layers[n.layer].push(n)});
  const layerOrder=Object.keys(layers).sort();
  const spacing=180,vSpacing=9000,margin=50;
  layerOrder.forEach((layer,i)=>{
    const layerNodes=layers[layer];
    const w=(layerNodes.length-1)*spacing;
    layerNodes.forEach((n,j)=>{n.x=margin+j*spacing-w/2;n.y=margin+i*vSpacing})
  });
  if(nodes.length>0){const xs=nodes.map(n=>n.x),ys=nodes.map(n=>n.y);
    const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const w=maxX-minX+100,h=maxY-minY+100;
    camera.scale=Math.min(canvas.width/devicePixelRatio/w,canvas.height/devicePixelRatio/h,1);
    camera.x=canvas.width/devicePixelRatio/2-(minX+maxX)/2*camera.scale;
    camera.y=canvas.height/devicePixelRatio/2-(minY+maxY)/2*camera.scale}
}

function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.save();ctx.translate(camera.x,camera.y);ctx.scale(camera.scale,camera.scale);
  edges.forEach(e=>{
    const s=nodes.find(n=>n.id===e.source),t=nodes.find(n=>n.id===e.target);
    if(!s||!t)return;
    ctx.strokeStyle=e.is_structural?"#F97316":"#38BDF8";ctx.lineWidth=e.is_structural?1.5:1;
    ctx.globalAlpha=0.6;ctx.beginPath();ctx.moveTo(s.x,s.y);ctx.lineTo(t.x,t.y);ctx.stroke();ctx.globalAlpha=1
  });
  nodes.forEach(n=>{
    ctx.fillStyle=n.is_structural?"#7C2D12":"#0F766E";
    ctx.beginPath();ctx.arc(n.x,n.y,6,0,2*Math.PI);ctx.fill();
    ctx.fillStyle="#F8FAFC";ctx.font="11px sans-serif";ctx.textAlign="center";
    const label=n.label.length>20?n.label.slice(0,18)+"…":n.label;
    ctx.fillText(label,n.x,n.y+18)
  });
  ctx.restore()
}

canvas.onmousedown=e=>{drag={x:e.offsetX,y:e.offsetY,camX:camera.x,camY:camera.y}};
canvas.onmousemove=e=>{if(drag){camera.x=drag.camX+e.offsetX-drag.x;camera.y=drag.camY+e.offsetY-drag.y;draw()}};
canvas.onmouseup=()=>drag=null;
canvas.onwheel=e=>{e.preventDefault();const factor=e.deltaY<0?1.1:0.9;camera.scale*=factor;draw()};
</script>
</body>
</html>
'''

if __name__ == "__main__":
    raise SystemExit(main())
