"""Lightweight, configurable-hop diabetes subgraph extraction for OptimusKG.

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
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

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
        ("discovered_hop", pa.int32()),
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
        ("hop_distance", pa.int32()),
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


def _empty_edge_table(path: Path, columns: Sequence[str]) -> pa.Table:
    """Return a projected empty table using a filter that cannot match an ID."""

    return pq.read_table(
        path,
        columns=list(columns),
        filters=[("from", "=", "__OPTIMUSKG_NO_SUCH_NODE__")],
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
    frontier: dict[str, set[str]],
    max_edges_per_type: int,
    discovered_hop: int,
    edge_types: Sequence[str] = EDGE_TYPES,
) -> dict[str, pa.Table]:
    """Extract one BFS layer of edges touching the typed frontier nodes."""

    result: dict[str, pa.Table] = {}
    edge_root = kg_root / "edges"

    if "disease_disease" in edge_types:
        columns = ["from", "to", "label", "relation", "undirected"]
        disease_ids = sorted(frontier.get("disease", set()))
        parts = []
        if disease_ids:
            parts.extend(
                [
                    _read_edge_table(
                        edge_root / "disease_disease.parquet",
                        columns,
                        "from",
                        disease_ids,
                    ),
                    _read_edge_table(
                        edge_root / "disease_disease.parquet",
                        columns,
                        "to",
                        disease_ids,
                    ),
                ]
            )
        table = _deduplicate_edges(pa.concat_tables(parts)) if parts else _empty_edge_table(
            edge_root / "disease_disease.parquet", columns
        )
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
        parts = []
        if frontier.get("disease"):
            parts.append(_read_edge_table(
                edge_root / "disease_gene.parquet", columns, "from",
                sorted(frontier["disease"]),
            ))
        if frontier.get("gene"):
            parts.append(_read_edge_table(
                edge_root / "disease_gene.parquet", columns, "to",
                sorted(frontier["gene"]),
            ))
        table = _deduplicate_edges(pa.concat_tables(parts)) if parts else _empty_edge_table(
            edge_root / "disease_gene.parquet", columns
        )
        result["disease_gene"] = _limit_table(
            table,
            max_edges_per_type,
            [("evidence_score", "descending"), ("evidence_count", "descending")],
        )

    if "disease_phenotype" in edge_types:
        columns = ["from", "to", "label", "relation", "undirected"]
        parts = []
        if frontier.get("disease"):
            parts.append(_read_edge_table(
                edge_root / "disease_phenotype.parquet", columns, "from",
                sorted(frontier["disease"]),
            ))
        if frontier.get("phenotype"):
            parts.append(_read_edge_table(
                edge_root / "disease_phenotype.parquet", columns, "to",
                sorted(frontier["phenotype"]),
            ))
        table = _deduplicate_edges(pa.concat_tables(parts)) if parts else _empty_edge_table(
            edge_root / "disease_phenotype.parquet", columns
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
        parts = []
        if frontier.get("disease"):
            parts.append(_read_edge_table(
                edge_root / "drug_disease.parquet", columns, "to",
                sorted(frontier["disease"]),
            ))
        if frontier.get("drug"):
            parts.append(_read_edge_table(
                edge_root / "drug_disease.parquet", columns, "from",
                sorted(frontier["drug"]),
            ))
        table = _deduplicate_edges(pa.concat_tables(parts)) if parts else _empty_edge_table(
            edge_root / "drug_disease.parquet", columns
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
        parts = []
        if frontier.get("disease"):
            parts.append(_read_edge_table(
                edge_root / "exposure_disease.parquet", columns, "to",
                sorted(frontier["disease"]),
            ))
        if frontier.get("exposure"):
            parts.append(_read_edge_table(
                edge_root / "exposure_disease.parquet", columns, "from",
                sorted(frontier["exposure"]),
            ))
        table = _deduplicate_edges(pa.concat_tables(parts)) if parts else _empty_edge_table(
            edge_root / "exposure_disease.parquet", columns
        )
        result["exposure_disease"] = _limit_table(
            table,
            max_edges_per_type,
            [("evidence_count", "descending"), ("from", "ascending")],
        )

    return result


def compact_edges(edge_tables: dict[str, list[tuple[int, pa.Table]]]) -> pa.Table:
    rows: list[dict[str, Any]] = []
    seen = set()
    for edge_type, hop_tables in edge_tables.items():
        for discovered_hop, table in hop_tables:
            for raw in table.to_pylist():
                key = (edge_type, raw.get("from"), raw.get("to"), raw.get("relation"))
                if key in seen:
                    continue
                seen.add(key)
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
                        "discovered_hop": discovered_hop,
                    }
                )
    return pa.Table.from_pylist(rows, schema=EDGE_SCHEMA)


def edge_node_types(edge_type: str, row: dict[str, Any]) -> tuple[tuple[str, str], tuple[str, str]]:
    endpoint_types = {
        "disease_disease": ("disease", "disease"),
        "disease_gene": ("disease", "gene"),
        "disease_phenotype": ("disease", "phenotype"),
        "drug_disease": ("drug", "disease"),
        "exposure_disease": ("exposure", "disease"),
    }
    from_type, to_type = endpoint_types[edge_type]
    return (from_type, row["from"]), (to_type, row["to"])


def traverse_edges(
    kg_root: Path,
    seed_ids: Sequence[str],
    hops: int,
    max_edges_per_type: int,
    edge_types: Sequence[str],
) -> tuple[dict[str, list[tuple[int, pa.Table]]], dict[tuple[str, str], int]]:
    distances = {("disease", seed_id): 0 for seed_id in seed_ids}
    frontier: dict[str, set[str]] = {"disease": set(seed_ids)}
    all_edges: dict[str, list[tuple[int, pa.Table]]] = defaultdict(list)

    for hop in range(1, hops + 1):
        layer = extract_edges(
            kg_root, frontier, max_edges_per_type, hop, edge_types
        )
        next_frontier: dict[str, set[str]] = defaultdict(set)
        for edge_type, table in layer.items():
            all_edges[edge_type].append((hop, table))
            for row in table.select(["from", "to"]).to_pylist():
                for typed_id in edge_node_types(edge_type, row):
                    if typed_id not in distances:
                        distances[typed_id] = hop
                        next_frontier[typed_id[0]].add(typed_id[1])
        frontier = next_frontier
        if not frontier:
            break
    return dict(all_edges), distances


def extract_nodes(
    kg_root: Path,
    distances: dict[tuple[str, str], int],
    seed_ids: Sequence[str],
) -> pa.Table:
    rows: list[dict[str, Any]] = []
    seed_set = set(seed_ids)
    for entity_type in ("disease", "gene", "phenotype", "drug", "exposure"):
        ids = sorted(node_id for (kind, node_id) in distances if kind == entity_type)
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
                    "hop_distance": distances[(entity_type, raw["id"])],
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
    hops: int,
    max_edges_per_type: int,
) -> None:
    """Fail before hand-off if the compact output violates demo guarantees."""

    node_ids = set(nodes["id"].to_pylist())
    if len(node_ids) != nodes.num_rows:
        raise RuntimeError("Extracted node IDs are not unique.")
    missing_seeds = set(seed_ids) - node_ids
    if missing_seeds:
        raise RuntimeError(f"Seed nodes are missing from output: {sorted(missing_seeds)}")

    for edge in edges.select(["from", "to"]).to_pylist():
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            raise RuntimeError(f"Edge endpoint is missing from nodes: {edge}")

    node_hops = nodes["hop_distance"].to_pylist()
    if node_hops and max(node_hops) > hops:
        raise RuntimeError("A node exceeds the requested hop distance.")

    if max_edges_per_type > 0:
        layer_counts: dict[tuple[str, int], int] = defaultdict(int)
        for edge in edges.select(["edge_type", "discovered_hop"]).to_pylist():
            layer_counts[(edge["edge_type"], edge["discovered_hop"])] += 1
        over_limit = {key: count for key, count in layer_counts.items()
                      if count > max_edges_per_type}
        if over_limit:
            raise RuntimeError(f"Per-hop/type edge limit was exceeded: {over_limit}")


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
    hops: int = 1,
    max_edges_per_type: int = 25,
    edge_types: Sequence[str] = EDGE_TYPES,
) -> dict[str, Any]:
    if max_edges_per_type < 0:
        raise ValueError("--max-edges-per-type must be >= 0 (0 means unlimited).")
    if hops < 0:
        raise ValueError("--hops must be >= 0.")
    unknown = sorted(set(edge_types) - set(EDGE_TYPES))
    if unknown:
        raise ValueError(f"Unknown edge type(s): {', '.join(unknown)}")

    kg_root = validate_kg_root(kg_root, edge_types)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = find_disease_candidates(kg_root, query, root_name)
    root = choose_root(candidates, root_id)
    seed_ids = [root["id"]]

    edge_tables, distances = traverse_edges(
        kg_root, seed_ids, hops, max_edges_per_type, edge_types
    )
    edges = compact_edges(edge_tables)
    nodes = extract_nodes(kg_root, distances, seed_ids)
    validate_subgraph(nodes, edges, seed_ids, hops, max_edges_per_type)

    pq.write_table(nodes, output_dir / "nodes.parquet", compression="zstd")
    pq.write_table(edges, output_dir / "edges.parquet", compression="zstd")
    _write_csv(nodes, output_dir / "nodes.csv")
    _write_csv(edges, output_dir / "edges.csv")
    write_candidates(candidates[:200], output_dir / "disease_candidates.csv")

    node_counts = defaultdict(int)
    for entity_type in nodes["entity_type"].to_pylist():
        node_counts[entity_type] += 1
    edge_counts: dict[str, int] = defaultdict(int)
    hop_edge_counts: dict[str, int] = defaultdict(int)
    for row in edges.select(["edge_type", "discovered_hop"]).to_pylist():
        edge_counts[row["edge_type"]] += 1
        hop_edge_counts[str(row["discovered_hop"])] += 1
    summary = {
        "kg_root": str(kg_root),
        "output_dir": str(output_dir),
        "query": query,
        "root": root,
        "hop_limit": hops,
        "max_edges_per_type": max_edges_per_type,
        "node_counts": dict(sorted(node_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
        "hop_edge_counts": dict(sorted(hop_edge_counts.items(), key=lambda item: int(item[0]))),
        "total_nodes": nodes.num_rows,
        "total_edges": edges.num_rows,
        "validation": {
            "requested_hop_limit_respected": True,
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
        description="Extract a compact N-hop diabetes subgraph from local OptimusKG Parquet files."
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
        "--hops",
        type=int,
        default=1,
        help="BFS hop count from the disease seed (default: 1; 0 writes only the seed).",
    )
    parser.add_argument(
        "--max-edges-per-type",
        type=int,
        default=25,
        help="Maximum retained edges per hop and relation type; 0 means unlimited.",
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
        hops=args.hops,
        max_edges_per_type=args.max_edges_per_type,
        edge_types=args.edge_types,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
