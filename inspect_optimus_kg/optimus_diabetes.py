"""Lightweight, one-hop diabetes subgraph extraction for OptimusKG.

This module deliberately uses PyArrow instead of loading OptimusKG into
NetworkX or a single in-memory dataframe.  Parquet column projection and
predicate pushdown keep the local demo small and make the same code usable on
a server with larger limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_KG_ROOT = Path(r"E:\code\data\knowledge\med\OptimusKG")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "diabetes_demo"
DEFAULT_QUERY = "diabetes"
DEFAULT_ROOT_NAME = "diabetes mellitus"

DISEASE_SEARCH_COLUMNS = [
    "id",
    "label",
    "properties.name",
    "properties.exact_synonyms",
    "properties.related_synonyms",
    "properties.narrow_synonyms",
    "properties.broad_synonyms",
    "properties.concept_names",
    "properties.snomed_full_names",
    "properties.umls_cui",
]

SEARCH_FIELDS = [
    "name",
    "exact_synonyms",
    "related_synonyms",
    "narrow_synonyms",
    "broad_synonyms",
    "concept_names",
    "snomed_full_names",
]

EDGE_TYPES = (
    "disease_disease",
    "disease_gene",
    "disease_phenotype",
    "drug_disease",
    "exposure_disease",
)

EDGE_SCHEMA = pa.schema(
    [
        ("from", pa.large_string()),
        ("to", pa.large_string()),
        ("label", pa.large_string()),
        ("relation", pa.large_string()),
        ("undirected", pa.bool_()),
        ("edge_type", pa.large_string()),
        ("evidence_score", pa.float64()),
        ("evidence_count", pa.int64()),
        ("highest_clinical_trial_phase", pa.float64()),
    ]
)

NODE_SCHEMA = pa.schema(
    [
        ("id", pa.large_string()),
        ("label", pa.large_string()),
        ("entity_type", pa.large_string()),
        ("name", pa.large_string()),
        ("symbol", pa.large_string()),
        ("is_seed", pa.bool_()),
    ]
)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required OptimusKG file not found: {path}")


def validate_kg_root(kg_root: Path, edge_types: Sequence[str] = EDGE_TYPES) -> Path:
    kg_root = kg_root.expanduser().resolve()
    _require_file(kg_root / "nodes" / "disease.parquet")
    for edge_type in edge_types:
        _require_file(kg_root / "edges" / f"{edge_type}.parquet")
    return kg_root


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _values(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return (str(item) for item in value if item is not None)
    return [str(value)]


def find_disease_candidates(
    kg_root: Path,
    query: str = DEFAULT_QUERY,
    root_name: str = DEFAULT_ROOT_NAME,
) -> list[dict[str, Any]]:
    """Find query matches without reading disease descriptions or other large fields."""

    disease_file = kg_root / "nodes" / "disease.parquet"
    table = pq.read_table(disease_file, columns=DISEASE_SEARCH_COLUMNS)
    query_norm = normalize_text(query)
    root_norm = normalize_text(root_name)
    candidates: list[dict[str, Any]] = []

    for row in table.to_pylist():
        matched_fields = []
        for field in SEARCH_FIELDS:
            if any(query_norm in normalize_text(value) for value in _values(row.get(field))):
                matched_fields.append(field)
        if not matched_fields:
            continue

        name_norm = normalize_text(row.get("name"))
        score = 0
        if name_norm == root_norm:
            score += 1000
        if name_norm == query_norm:
            score += 500
        if "name" in matched_fields:
            score += 100
        if str(row["id"]).startswith("EFO_"):
            # This export's EFO diabetes mellitus node carries the usable hierarchy.
            score += 20
        if str(row["id"]).startswith("MONDO_"):
            score += 10

        candidates.append(
            {
                "id": row["id"],
                "label": row.get("label"),
                "name": row.get("name"),
                "umls_cui": row.get("umls_cui"),
                "matched_fields": ",".join(matched_fields),
                "selection_score": score,
            }
        )

    return sorted(
        candidates,
        key=lambda row: (-row["selection_score"], normalize_text(row["name"]), row["id"]),
    )


def choose_root(
    candidates: Sequence[dict[str, Any]], root_id: str | None = None
) -> dict[str, Any]:
    if root_id:
        for candidate in candidates:
            if candidate["id"] == root_id:
                return dict(candidate)
        raise ValueError(
            f"--root-id {root_id!r} did not match the query. "
            "Run 01_find_diabetes.py to inspect candidates."
        )
    if not candidates:
        raise ValueError("No disease candidate matched the requested query.")
    return dict(candidates[0])


def _read_edge_table(
    path: Path,
    columns: Sequence[str],
    endpoint: str,
    disease_ids: Sequence[str],
) -> pa.Table:
    return pq.read_table(
        path,
        columns=list(columns),
        filters=[(endpoint, "in", list(disease_ids))],
        use_threads=True,
    )


def _deduplicate_edges(table: pa.Table) -> pa.Table:
    rows = {}
    for row in table.to_pylist():
        key = (row.get("from"), row.get("to"), row.get("relation"))
        rows.setdefault(key, row)
    return pa.Table.from_pylist(list(rows.values()), schema=table.schema)


def _limit_table(
    table: pa.Table,
    limit: int,
    sort_by: Sequence[tuple[str, str]] = (),
) -> pa.Table:
    if sort_by and table.num_rows:
        table = table.sort_by(list(sort_by))
    return table if limit <= 0 else table.slice(0, limit)


def extract_edges(
    kg_root: Path,
    disease_ids: Sequence[str],
    max_edges_per_type: int,
    edge_types: Sequence[str] = EDGE_TYPES,
) -> dict[str, pa.Table]:
    """Extract only direct edges touching the disease seed(s)."""

    result: dict[str, pa.Table] = {}
    edge_root = kg_root / "edges"

    if "disease_disease" in edge_types:
        columns = ["from", "to", "label", "relation", "undirected"]
        outgoing = _read_edge_table(
            edge_root / "disease_disease.parquet", columns, "from", disease_ids
        )
        incoming = _read_edge_table(
            edge_root / "disease_disease.parquet", columns, "to", disease_ids
        )
        table = _deduplicate_edges(pa.concat_tables([outgoing, incoming]))
        result["disease_disease"] = _limit_table(
            table, max_edges_per_type, [("from", "ascending"), ("to", "ascending")]
        )

    if "disease_gene" in edge_types:
        columns = [
            "from",
            "to",
            "label",
            "relation",
            "undirected",
            "properties.evidence_score",
            "properties.evidence_count",
        ]
        table = _read_edge_table(
            edge_root / "disease_gene.parquet", columns, "from", disease_ids
        )
        result["disease_gene"] = _limit_table(
            table,
            max_edges_per_type,
            [("evidence_score", "descending"), ("evidence_count", "descending")],
        )

    if "disease_phenotype" in edge_types:
        columns = ["from", "to", "label", "relation", "undirected"]
        table = _read_edge_table(
            edge_root / "disease_phenotype.parquet", columns, "from", disease_ids
        )
        result["disease_phenotype"] = _limit_table(
            table, max_edges_per_type, [("to", "ascending")]
        )

    if "drug_disease" in edge_types:
        columns = [
            "from",
            "to",
            "label",
            "relation",
            "undirected",
            "properties.highest_clinical_trial_phase",
        ]
        table = _read_edge_table(
            edge_root / "drug_disease.parquet", columns, "to", disease_ids
        )
        result["drug_disease"] = _limit_table(
            table,
            max_edges_per_type,
            [("highest_clinical_trial_phase", "descending"), ("from", "ascending")],
        )

    if "exposure_disease" in edge_types:
        columns = [
            "from",
            "to",
            "label",
            "relation",
            "undirected",
            "properties.evidence_count",
        ]
        table = _read_edge_table(
            edge_root / "exposure_disease.parquet", columns, "to", disease_ids
        )
        result["exposure_disease"] = _limit_table(
            table,
            max_edges_per_type,
            [("evidence_count", "descending"), ("from", "ascending")],
        )

    return result


def compact_edges(edge_tables: dict[str, pa.Table]) -> pa.Table:
    rows: list[dict[str, Any]] = []
    for edge_type, table in edge_tables.items():
        for raw in table.to_pylist():
            rows.append(
                {
                    "from": raw.get("from"),
                    "to": raw.get("to"),
                    "label": raw.get("label"),
                    "relation": raw.get("relation"),
                    "undirected": raw.get("undirected"),
                    "edge_type": edge_type,
                    "evidence_score": raw.get("evidence_score"),
                    "evidence_count": raw.get("evidence_count"),
                    "highest_clinical_trial_phase": raw.get(
                        "highest_clinical_trial_phase"
                    ),
                }
            )
    return pa.Table.from_pylist(rows, schema=EDGE_SCHEMA)


def collect_node_ids(
    seed_ids: Sequence[str], edge_tables: dict[str, pa.Table]
) -> dict[str, set[str]]:
    node_ids: dict[str, set[str]] = defaultdict(set)
    node_ids["disease"].update(seed_ids)
    for edge_type, table in edge_tables.items():
        for row in table.select(["from", "to"]).to_pylist():
            if edge_type == "disease_disease":
                node_ids["disease"].update([row["from"], row["to"]])
            elif edge_type == "disease_gene":
                node_ids["gene"].add(row["to"])
            elif edge_type == "disease_phenotype":
                node_ids["phenotype"].add(row["to"])
            elif edge_type == "drug_disease":
                node_ids["drug"].add(row["from"])
            elif edge_type == "exposure_disease":
                node_ids["exposure"].add(row["from"])
    return node_ids


def extract_nodes(
    kg_root: Path,
    node_ids: dict[str, set[str]],
    seed_ids: Sequence[str],
) -> pa.Table:
    rows: list[dict[str, Any]] = []
    seed_set = set(seed_ids)
    for entity_type in ("disease", "gene", "phenotype", "drug", "exposure"):
        ids = sorted(node_ids.get(entity_type, set()))
        if not ids:
            continue
        columns = ["id", "label", "properties.name"]
        if entity_type == "gene":
            columns.append("properties.symbol")
        table = pq.read_table(
            kg_root / "nodes" / f"{entity_type}.parquet",
            columns=columns,
            filters=[("id", "in", ids)],
            use_threads=True,
        )
        for raw in table.to_pylist():
            rows.append(
                {
                    "id": raw["id"],
                    "label": raw.get("label"),
                    "entity_type": entity_type,
                    "name": raw.get("name"),
                    "symbol": raw.get("symbol"),
                    "is_seed": raw["id"] in seed_set,
                }
            )
    return pa.Table.from_pylist(rows, schema=NODE_SCHEMA)


def _write_csv(table: pa.Table, path: Path) -> None:
    rows = table.to_pylist()
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table.column_names)
        writer.writeheader()
        writer.writerows(rows)


def validate_subgraph(
    nodes: pa.Table,
    edges: pa.Table,
    seed_ids: Sequence[str],
    edge_counts: dict[str, int],
    max_edges_per_type: int,
) -> None:
    """Fail before hand-off if the compact output violates demo guarantees."""

    node_ids = set(nodes["id"].to_pylist())
    if len(node_ids) != nodes.num_rows:
        raise RuntimeError("Extracted node IDs are not unique.")
    missing_seeds = set(seed_ids) - node_ids
    if missing_seeds:
        raise RuntimeError(f"Seed nodes are missing from output: {sorted(missing_seeds)}")

    seed_set = set(seed_ids)
    for edge in edges.select(["from", "to"]).to_pylist():
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            raise RuntimeError(f"Edge endpoint is missing from nodes: {edge}")
        if edge["from"] not in seed_set and edge["to"] not in seed_set:
            raise RuntimeError(f"Non-one-hop edge found: {edge}")

    if max_edges_per_type > 0:
        over_limit = {
            name: count
            for name, count in edge_counts.items()
            if count > max_edges_per_type
        }
        if over_limit:
            raise RuntimeError(f"Per-type edge limit was exceeded: {over_limit}")


def write_candidates(candidates: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(candidates))
    _write_csv(table, path)


def run_extraction(
    kg_root: Path,
    output_dir: Path,
    query: str = DEFAULT_QUERY,
    root_name: str = DEFAULT_ROOT_NAME,
    root_id: str | None = None,
    max_edges_per_type: int = 25,
    edge_types: Sequence[str] = EDGE_TYPES,
) -> dict[str, Any]:
    if max_edges_per_type < 0:
        raise ValueError("--max-edges-per-type must be >= 0 (0 means unlimited).")
    unknown = sorted(set(edge_types) - set(EDGE_TYPES))
    if unknown:
        raise ValueError(f"Unknown edge type(s): {', '.join(unknown)}")

    kg_root = validate_kg_root(kg_root, edge_types)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = find_disease_candidates(kg_root, query, root_name)
    root = choose_root(candidates, root_id)
    seed_ids = [root["id"]]

    edge_tables = extract_edges(kg_root, seed_ids, max_edges_per_type, edge_types)
    edges = compact_edges(edge_tables)
    node_ids = collect_node_ids(seed_ids, edge_tables)
    nodes = extract_nodes(kg_root, node_ids, seed_ids)
    edge_counts = {name: table.num_rows for name, table in edge_tables.items()}
    validate_subgraph(nodes, edges, seed_ids, edge_counts, max_edges_per_type)

    pq.write_table(nodes, output_dir / "nodes.parquet", compression="zstd")
    pq.write_table(edges, output_dir / "edges.parquet", compression="zstd")
    _write_csv(nodes, output_dir / "nodes.csv")
    _write_csv(edges, output_dir / "edges.csv")
    write_candidates(candidates[:200], output_dir / "disease_candidates.csv")

    node_counts = defaultdict(int)
    for entity_type in nodes["entity_type"].to_pylist():
        node_counts[entity_type] += 1
    summary = {
        "kg_root": str(kg_root),
        "output_dir": str(output_dir),
        "query": query,
        "root": root,
        "hop_limit": 1,
        "max_edges_per_type": max_edges_per_type,
        "node_counts": dict(sorted(node_counts.items())),
        "edge_counts": edge_counts,
        "total_nodes": nodes.num_rows,
        "total_edges": edges.num_rows,
        "validation": {
            "one_hop_only": True,
            "all_edge_endpoints_resolved": True,
            "unique_node_ids": True,
            "edge_limits_respected": True,
        },
        "artifact_bytes_excluding_summary": sum(
            path.stat().st_size for path in output_dir.iterdir() if path.is_file()
            and path.name != "summary.json"
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a compact one-hop diabetes subgraph from local OptimusKG Parquet files."
    )
    parser.add_argument("--kg-root", type=Path, default=DEFAULT_KG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--root-name", default=DEFAULT_ROOT_NAME)
    parser.add_argument(
        "--root-id",
        help="Explicit disease seed ID. Default: auto-select the best exact-name match.",
    )
    parser.add_argument(
        "--max-edges-per-type",
        type=int,
        default=25,
        help="Maximum retained edges for each relation file; 0 means unlimited.",
    )
    parser.add_argument(
        "--edge-types",
        nargs="+",
        choices=EDGE_TYPES,
        default=list(EDGE_TYPES),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_extraction(
        kg_root=args.kg_root,
        output_dir=args.output_dir,
        query=args.query,
        root_name=args.root_name,
        root_id=args.root_id,
        max_edges_per_type=args.max_edges_per_type,
        edge_types=args.edge_types,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
