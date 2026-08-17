#!/usr/bin/env python3
"""Build a deterministic dual-keyword BFS PNet from the Gold SQLite graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml


ALGORITHM_NAME = "dual_keyword_bfs_frontier_bridge"
ALGORITHM_VERSION = "1.0"
REQUIRED_TABLES = {
    "entities",
    "entity_aliases",
    "entity_external_ids",
    "assertions",
    "relation_types",
    "raw_assertions",
}

NODE_FIELDS = (
    "node_id",
    "entity_id",
    "layer",
    "node_type",
    "display_name",
    "is_fallback",
    "source",
    "external_id",
    "original_node_id",
    "description",
    "side",
    "bfs_depth",
    "matched_keywords",
    "is_structural",
)
EDGE_FIELDS = (
    "source_node_id",
    "target_node_id",
    "relation_type",
    "knowledge_source",
    "confidence",
    "is_fallback",
    "evidence_relation_ids",
    "enabled",
    "edge_kind",
    "is_structural",
    "original_source_entity_id",
    "original_target_entity_id",
    "original_direction",
    "traversal_direction",
)
MATCH_FIELDS = (
    "side",
    "keyword",
    "normalized_keyword",
    "entity_id",
    "display_name",
    "matched_field",
    "matched_text",
    "match_method",
)
OCCURRENCE_FIELDS = (
    "side",
    "node_id",
    "entity_id",
    "bfs_depth",
    "is_seed",
    "is_carry",
    "parent_node_ids",
    "parent_relation_ids",
)
REJECTED_FIELDS = (
    "side",
    "source_entity_id",
    "target_entity_id",
    "relation_id",
    "source_depth",
    "target_depth",
    "rejection_reason",
)


class BuildError(RuntimeError):
    """Raised when a requested PNet cannot be built completely and safely."""


@dataclass(frozen=True)
class BuildConfig:
    sqlite: Path
    output_dir: Path
    text_s: tuple[str, ...]
    text_t: tuple[str, ...]
    source_max_hops: int = 1
    target_max_hops: int = 1
    traversal_mode: str = "undirected"
    dead_end_policy: str = "structural_carry"
    max_nodes: int = 100_000
    max_edges: int = 1_000_000
    max_bridge_edges: int = 500_000
    kg_name: str = "medical_kg_gold"
    kg_version: str | None = None

    def checked(self) -> BuildConfig:
        if not self.text_s or not self.text_t:
            raise BuildError("text_s 和 text_t 均必须是非空关键词列表")
        if any(not normalize_text(value) for value in (*self.text_s, *self.text_t)):
            raise BuildError("关键词不能是空白文本")
        if self.source_max_hops < 0 or self.target_max_hops < 0:
            raise BuildError("BFS 最大跳数不能小于 0")
        if self.traversal_mode not in {"undirected", "directed"}:
            raise BuildError("traversal_mode 必须是 undirected 或 directed")
        if self.dead_end_policy != "structural_carry":
            raise BuildError("当前构建器仅支持保证关键词实体不丢失的 structural_carry 策略")
        if min(self.max_nodes, self.max_edges, self.max_bridge_edges) < 1:
            raise BuildError("max_nodes/max_edges/max_bridge_edges 均必须大于 0")
        if not self.sqlite.is_file():
            raise BuildError(f"Gold SQLite 文件不存在：{self.sqlite}")
        return self


@dataclass(frozen=True)
class Entity:
    entity_id: str
    display_name: str
    entity_type: str
    description: str
    aliases: tuple[str, ...] = ()
    external_id: str = ""


@dataclass(frozen=True)
class Relation:
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    confidence: float


@dataclass(frozen=True)
class Match:
    side: str
    keyword: str
    normalized_keyword: str
    entity_id: str
    display_name: str
    matched_field: str
    matched_text: str
    match_method: str


@dataclass
class Node:
    node_id: str
    entity_id: str
    layer: str
    node_type: str
    display_name: str
    is_fallback: bool
    source: str
    external_id: str = ""
    original_node_id: str = ""
    description: str = ""
    side: str = ""
    bfs_depth: int = 0
    matched_keywords: str = ""
    is_structural: bool = False


@dataclass
class Edge:
    source_node_id: str
    target_node_id: str
    relation_type: str
    knowledge_source: str
    confidence: float
    is_fallback: bool
    evidence_relation_ids: str = ""
    enabled: bool = True
    edge_kind: str = "kg_bfs"
    is_structural: bool = False
    original_source_entity_id: str = ""
    original_target_entity_id: str = ""
    original_direction: str = ""
    traversal_direction: str = ""


@dataclass
class Occurrence:
    side: str
    node_id: str
    entity_id: str
    bfs_depth: int
    is_seed: bool
    is_carry: bool
    parent_node_ids: str = ""
    parent_relation_ids: str = ""


@dataclass
class SideResult:
    side: str
    max_hops: int
    distances: dict[str, int]
    parents: dict[str, set[tuple[str, str]]]
    nodes: dict[str, Node] = field(default_factory=dict)
    occurrences: list[Occurrence] = field(default_factory=list)
    edges: dict[tuple[str, str], Edge] = field(default_factory=dict)
    frontier_node_ids: list[str] = field(default_factory=list)
    carry_node_count: int = 0
    carry_edge_count: int = 0


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    node_count: int
    edge_count: int
    source_match_count: int
    target_match_count: int
    bridge_edge_count: int
    manifest: Mapping[str, Any]


def normalize_text(value: str) -> str:
    """Apply the fixed matching normalization described by requirement.md."""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"[\u2010-\u2015_\-]+", " ", text)
    text = re.sub(r"[,;:/\\|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _node_id(side: str, depth: int, entity_id: str) -> str:
    prefix = "s" if side == "source" else "t"
    return f"{prefix}::d{depth:03d}::{quote(entity_id, safe='-._~')}"


def _carry_ids(side: str, anchor_entity_id: str, depth: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{side}\0{anchor_entity_id}".encode()).hexdigest()[:16]
    prefix = "s" if side == "source" else "t"
    entity_id = f"STRUCTURAL_CARRY::{side}::{digest}::d{depth:03d}"
    return f"{prefix}::d{depth:03d}::carry::{digest}", entity_id


def _layer_id(side: str, depth: int) -> str:
    return f"{side}_d{depth:03d}"


def _sqlite_connection(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _require_schema(connection: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    missing = sorted(REQUIRED_TABLES - tables)
    if missing:
        raise BuildError(f"SQLite 缺少 Gold PNet 所需表：{', '.join(missing)}")


def _load_entities(connection: sqlite3.Connection) -> dict[str, Entity]:
    aliases: dict[str, list[str]] = defaultdict(list)
    for row in connection.execute(
        "SELECT entity_id, alias FROM entity_aliases ORDER BY entity_id, alias"
    ):
        aliases[str(row["entity_id"])].append(str(row["alias"]))

    external_ids: dict[str, str] = {}
    for row in connection.execute(
        "SELECT entity_id, normalized_id, is_primary FROM entity_external_ids "
        "ORDER BY entity_id, is_primary DESC, normalized_id"
    ):
        external_ids.setdefault(str(row["entity_id"]), str(row["normalized_id"]))

    entities: dict[str, Entity] = {}
    query = (
        "SELECT entity_id, canonical_name, entity_type, COALESCE(description, '') description "
        "FROM entities ORDER BY entity_id"
    )
    for row in connection.execute(query):
        entity_id = str(row["entity_id"])
        entities[entity_id] = Entity(
            entity_id=entity_id,
            display_name=str(row["canonical_name"]),
            entity_type=str(row["entity_type"]),
            description=str(row["description"]),
            aliases=tuple(aliases.get(entity_id, ())),
            external_id=external_ids.get(entity_id, ""),
        )
    return entities


def _load_relations(connection: sqlite3.Connection) -> list[Relation]:
    query = """
        SELECT a.assertion_id,
               a.subject_entity_id,
               a.object_entity_id,
               rt.canonical_name relation_type,
               COALESCE(ra.llm_confidence, 1.0) confidence
        FROM assertions a
        JOIN relation_types rt ON rt.relation_id = a.canonical_relation_id
        JOIN raw_assertions ra ON ra.raw_assertion_id = a.raw_assertion_id
        ORDER BY a.subject_entity_id, a.object_entity_id, a.assertion_id
    """
    return [
        Relation(
            relation_id=str(row["assertion_id"]),
            source_entity_id=str(row["subject_entity_id"]),
            target_entity_id=str(row["object_entity_id"]),
            relation_type=str(row["relation_type"]),
            confidence=max(0.0, min(1.0, float(row["confidence"]))),
        )
        for row in connection.execute(query)
    ]


def _match_entities(
    entities: Mapping[str, Entity], keywords: Sequence[str], side: str
) -> tuple[list[Match], set[str]]:
    normalized_keywords = [(keyword, normalize_text(keyword)) for keyword in keywords]
    matches: list[Match] = []
    matched_ids: set[str] = set()
    for entity_id in sorted(entities):
        entity = entities[entity_id]
        fields: list[tuple[str, str]] = [("display_name", entity.display_name)]
        fields.extend(("aliases", alias) for alias in entity.aliases)
        if entity.description:
            fields.append(("description", entity.description))
        for keyword, normalized_keyword in normalized_keywords:
            for field_name, field_text in fields:
                if normalized_keyword in normalize_text(field_text):
                    matches.append(
                        Match(
                            side=side,
                            keyword=keyword,
                            normalized_keyword=normalized_keyword,
                            entity_id=entity_id,
                            display_name=entity.display_name,
                            matched_field=field_name,
                            matched_text=field_text,
                            match_method="normalized_substring",
                        )
                    )
                    matched_ids.add(entity_id)
            if normalized_keyword == normalize_text(entity_id):
                matches.append(
                    Match(
                        side=side,
                        keyword=keyword,
                        normalized_keyword=normalized_keyword,
                        entity_id=entity_id,
                        display_name=entity.display_name,
                        matched_field="entity_id",
                        matched_text=entity_id,
                        match_method="normalized_exact_id",
                    )
                )
                matched_ids.add(entity_id)
    matches.sort(key=lambda item: tuple(str(value) for value in asdict(item).values()))
    return matches, matched_ids


def _adjacency(
    relations: Sequence[Relation], side: str, traversal_mode: str
) -> dict[str, list[tuple[str, Relation]]]:
    adjacency: dict[str, list[tuple[str, Relation]]] = defaultdict(list)
    for relation in relations:
        source = relation.source_entity_id
        target = relation.target_entity_id
        if traversal_mode == "undirected":
            adjacency[source].append((target, relation))
            if target != source:
                adjacency[target].append((source, relation))
        elif side == "source":
            adjacency[source].append((target, relation))
        else:
            adjacency[target].append((source, relation))
    for entity_id in adjacency:
        adjacency[entity_id].sort(key=lambda item: (item[0], item[1].relation_id))
    return adjacency


def _bfs(
    seeds: set[str],
    relations: Sequence[Relation],
    side: str,
    traversal_mode: str,
    max_hops: int,
) -> tuple[dict[str, int], dict[str, set[tuple[str, str]]]]:
    adjacency = _adjacency(relations, side, traversal_mode)
    distances = {entity_id: 0 for entity_id in sorted(seeds)}
    current = sorted(seeds)
    for depth in range(max_hops):
        next_entities: set[str] = set()
        for entity_id in current:
            for neighbor_id, _relation in adjacency.get(entity_id, ()):
                if neighbor_id not in distances:
                    distances[neighbor_id] = depth + 1
                    next_entities.add(neighbor_id)
        current = sorted(next_entities)
        if not current:
            break

    parents: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for parent_id in sorted(distances):
        parent_depth = distances[parent_id]
        if parent_depth >= max_hops:
            continue
        for child_id, relation in adjacency.get(parent_id, ()):
            if distances.get(child_id) == parent_depth + 1:
                parents[child_id].add((parent_id, relation.relation_id))
    return distances, parents


def _real_node(
    entity: Entity,
    side: str,
    depth: int,
    keywords_by_entity: Mapping[str, set[str]],
    kg_name: str,
) -> Node:
    return Node(
        node_id=_node_id(side, depth, entity.entity_id),
        entity_id=entity.entity_id,
        layer=_layer_id(side, depth),
        node_type=entity.entity_type,
        display_name=entity.display_name or entity.entity_id,
        is_fallback=False,
        source=kg_name,
        external_id=entity.external_id,
        description=entity.description,
        side=side,
        bfs_depth=depth,
        matched_keywords="|".join(sorted(keywords_by_entity.get(entity.entity_id, ()))),
        is_structural=False,
    )


def _accepted_orientation(
    relation: Relation,
    distances: Mapping[str, int],
    side: str,
    traversal_mode: str,
) -> tuple[str, str, str] | None:
    source = relation.source_entity_id
    target = relation.target_entity_id
    source_depth = distances.get(source)
    target_depth = distances.get(target)
    if source_depth is None or target_depth is None or abs(source_depth - target_depth) != 1:
        return None
    if side == "source":
        shallow, deep = (
            (source, target) if source_depth < target_depth else (target, source)
        )
        if traversal_mode == "directed" and (shallow, deep) != (source, target):
            return None
        traversal = "forward" if (shallow, deep) == (source, target) else "reverse"
        return shallow, deep, traversal

    deep, shallow = (
        (source, target) if source_depth > target_depth else (target, source)
    )
    if traversal_mode == "directed" and (deep, shallow) != (source, target):
        return None
    traversal = "forward" if (deep, shallow) == (source, target) else "reverse"
    return deep, shallow, traversal


def _build_kg_edges(
    side_result: SideResult,
    relations: Sequence[Relation],
    traversal_mode: str,
    kg_name: str,
) -> None:
    grouped: dict[tuple[str, str], list[tuple[Relation, str]]] = defaultdict(list)
    for relation in relations:
        orientation = _accepted_orientation(
            relation, side_result.distances, side_result.side, traversal_mode
        )
        if orientation is None:
            continue
        source_entity_id, target_entity_id, traversal = orientation
        source_depth = side_result.distances[source_entity_id]
        target_depth = side_result.distances[target_entity_id]
        source_node = _node_id(side_result.side, source_depth, source_entity_id)
        target_node = _node_id(side_result.side, target_depth, target_entity_id)
        grouped[(source_node, target_node)].append((relation, traversal))

    for pair in sorted(grouped):
        evidence = grouped[pair]
        relation_types = sorted({item[0].relation_type for item in evidence})
        relation_ids = sorted({item[0].relation_id for item in evidence})
        original_sources = sorted({item[0].source_entity_id for item in evidence})
        original_targets = sorted({item[0].target_entity_id for item in evidence})
        directions = sorted(
            {
                f"{item[0].source_entity_id}->{item[0].target_entity_id}"
                for item in evidence
            }
        )
        traversals = sorted({item[1] for item in evidence})
        side_result.edges[pair] = Edge(
            source_node_id=pair[0],
            target_node_id=pair[1],
            relation_type="|".join(relation_types),
            knowledge_source=kg_name,
            confidence=max(item[0].confidence for item in evidence),
            is_fallback=False,
            evidence_relation_ids="|".join(relation_ids),
            edge_kind="kg_bfs",
            is_structural=False,
            original_source_entity_id="|".join(original_sources),
            original_target_entity_id="|".join(original_targets),
            original_direction="|".join(directions),
            traversal_direction="|".join(traversals),
        )


def _make_side(
    side: str,
    max_hops: int,
    seeds: set[str],
    matches: Sequence[Match],
    entities: Mapping[str, Entity],
    relations: Sequence[Relation],
    traversal_mode: str,
    kg_name: str,
) -> SideResult:
    distances, parents = _bfs(seeds, relations, side, traversal_mode, max_hops)
    result = SideResult(side=side, max_hops=max_hops, distances=distances, parents=parents)
    keywords_by_entity: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        keywords_by_entity[match.entity_id].add(match.keyword)

    for entity_id in sorted(distances, key=lambda value: (distances[value], value)):
        depth = distances[entity_id]
        node = _real_node(entities[entity_id], side, depth, keywords_by_entity, kg_name)
        result.nodes[node.node_id] = node
        parent_pairs = sorted(parents.get(entity_id, ()))
        result.occurrences.append(
            Occurrence(
                side=side,
                node_id=node.node_id,
                entity_id=entity_id,
                bfs_depth=depth,
                is_seed=depth == 0,
                is_carry=False,
                parent_node_ids="|".join(
                    sorted({_node_id(side, depth - 1, item[0]) for item in parent_pairs})
                ),
                parent_relation_ids="|".join(sorted({item[1] for item in parent_pairs})),
            )
        )

    _build_kg_edges(result, relations, traversal_mode, kg_name)
    entities_with_children = {
        parent_id for child_parents in parents.values() for parent_id, _relation_id in child_parents
    }
    for anchor_id in sorted(distances, key=lambda value: (distances[value], value)):
        start_depth = distances[anchor_id]
        if start_depth >= max_hops or anchor_id in entities_with_children:
            continue
        previous_node_id = _node_id(side, start_depth, anchor_id)
        for depth in range(start_depth + 1, max_hops + 1):
            carry_node_id, carry_entity_id = _carry_ids(side, anchor_id, depth)
            carry_node = Node(
                node_id=carry_node_id,
                entity_id=carry_entity_id,
                layer=_layer_id(side, depth),
                node_type="structural_carry",
                display_name=f"Carry: {entities[anchor_id].display_name}",
                is_fallback=True,
                source="dual_bfs_algorithm",
                original_node_id=previous_node_id,
                description="Structural layer-alignment node; not medical knowledge.",
                side=side,
                bfs_depth=depth,
                is_structural=True,
            )
            result.nodes[carry_node_id] = carry_node
            result.occurrences.append(
                Occurrence(
                    side=side,
                    node_id=carry_node_id,
                    entity_id=carry_entity_id,
                    bfs_depth=depth,
                    is_seed=False,
                    is_carry=True,
                    parent_node_ids=previous_node_id,
                )
            )
            pair = (
                (previous_node_id, carry_node_id)
                if side == "source"
                else (carry_node_id, previous_node_id)
            )
            result.edges[pair] = Edge(
                source_node_id=pair[0],
                target_node_id=pair[1],
                relation_type="structural_carry",
                knowledge_source="dual_bfs_algorithm",
                confidence=1.0,
                is_fallback=True,
                edge_kind="structural_carry",
                is_structural=True,
                traversal_direction="structural",
            )
            previous_node_id = carry_node_id
            result.carry_node_count += 1
            result.carry_edge_count += 1

    result.frontier_node_ids = sorted(
        node.node_id for node in result.nodes.values() if node.bfs_depth == max_hops
    )
    return result


def _rejected_edges(
    side: str,
    max_hops: int,
    distances: Mapping[str, int],
    relations: Sequence[Relation],
    traversal_mode: str,
) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for relation in relations:
        source_depth = distances.get(relation.source_entity_id)
        target_depth = distances.get(relation.target_entity_id)
        reason = ""
        if relation.source_entity_id == relation.target_entity_id and source_depth is not None:
            reason = "self_loop"
        elif source_depth is not None and target_depth is not None:
            if source_depth == target_depth:
                reason = "same_layer"
            elif abs(source_depth - target_depth) > 1:
                reason = "depth_gap"
            elif _accepted_orientation(relation, distances, side, traversal_mode) is None:
                reason = "backward"
        elif traversal_mode == "undirected":
            visited_depth = source_depth if source_depth is not None else target_depth
            if visited_depth == max_hops:
                reason = "outside_hop_limit"
        elif side == "source" and source_depth == max_hops and target_depth is None:
            reason = "outside_hop_limit"
        elif side == "target" and target_depth == max_hops and source_depth is None:
            reason = "outside_hop_limit"
        if reason:
            rejected.append(
                {
                    "side": side,
                    "source_entity_id": relation.source_entity_id,
                    "target_entity_id": relation.target_entity_id,
                    "relation_id": relation.relation_id,
                    "source_depth": "" if source_depth is None else source_depth,
                    "target_depth": "" if target_depth is None else target_depth,
                    "rejection_reason": reason,
                }
            )
    rejected.sort(
        key=lambda row: (
            row["side"],
            row["source_entity_id"],
            row["target_entity_id"],
            row["relation_id"],
            row["rejection_reason"],
        )
    )
    return rejected


def _layers(source_hops: int, target_hops: int) -> list[dict[str, Any]]:
    layers = [
        {"id": _layer_id("source", depth), "order": depth}
        for depth in range(source_hops + 1)
    ]
    order = source_hops + 1
    for depth in range(target_hops, -1, -1):
        layers.append({"id": _layer_id("target", depth), "order": order})
        order += 1
    return layers


def _validate(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    layers: Sequence[Mapping[str, Any]],
    source_frontier: Sequence[str],
    target_frontier: Sequence[str],
) -> None:
    errors: list[str] = []
    node_by_id = {node.node_id: node for node in nodes}
    if len(node_by_id) != len(nodes):
        errors.append("node_id 非全局唯一")
    layer_order = {str(layer["id"]): int(layer["order"]) for layer in layers}
    widths: dict[str, int] = defaultdict(int)
    for node in nodes:
        widths[node.layer] += 1
        if not node.display_name:
            errors.append(f"节点显示名为空：{node.node_id}")
        if node.layer not in layer_order:
            errors.append(f"节点引用未知层：{node.node_id} -> {node.layer}")
    empty_layers = [layer_id for layer_id in layer_order if widths[layer_id] == 0]
    if empty_layers:
        errors.append(f"存在空层：{', '.join(empty_layers)}")

    pairs: set[tuple[str, str]] = set()
    outgoing: dict[str, list[str]] = defaultdict(list)
    bridge_count = 0
    for edge in edges:
        pair = (edge.source_node_id, edge.target_node_id)
        if pair in pairs:
            errors.append(f"重复 source-target 边：{pair[0]} -> {pair[1]}")
        pairs.add(pair)
        if edge.source_node_id not in node_by_id or edge.target_node_id not in node_by_id:
            errors.append(f"边引用不存在的端点：{pair[0]} -> {pair[1]}")
            continue
        source = node_by_id[edge.source_node_id]
        target = node_by_id[edge.target_node_id]
        if source.node_id == target.node_id:
            errors.append(f"PNet 自环：{source.node_id}")
        if layer_order[target.layer] != layer_order[source.layer] + 1:
            errors.append(f"非相邻层边：{pair[0]} -> {pair[1]}")
        if edge.edge_kind == "kg_bfs" and not edge.evidence_relation_ids:
            errors.append(f"KG 边缺少 evidence_relation_ids：{pair[0]} -> {pair[1]}")
        if edge.edge_kind in {"structural_bridge", "structural_carry"}:
            if not edge.is_structural or edge.knowledge_source != "dual_bfs_algorithm":
                errors.append(f"人工结构边标记不完整：{pair[0]} -> {pair[1]}")
        if edge.edge_kind == "structural_bridge":
            bridge_count += 1
        outgoing[edge.source_node_id].append(edge.target_node_id)

    expected_bridges = len(source_frontier) * len(target_frontier)
    if bridge_count != expected_bridges:
        errors.append(f"桥边数量错误：expected={expected_bridges}, actual={bridge_count}")

    final_order = max(layer_order.values())
    can_reach_final = {
        node.node_id for node in nodes if layer_order.get(node.layer) == final_order
    }
    for order in range(final_order - 1, -1, -1):
        for node in nodes:
            if layer_order.get(node.layer) != order:
                continue
            if any(target in can_reach_final for target in outgoing.get(node.node_id, ())):
                can_reach_final.add(node.node_id)
    unreachable = sorted(set(node_by_id) - can_reach_final)
    if unreachable:
        preview = ", ".join(unreachable[:5])
        errors.append(f"{len(unreachable)} 个节点不能到达 target_d000（示例：{preview}）")
    if errors:
        raise BuildError("PNet 验证失败：\n- " + "\n- ".join(errors[:20]))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _tsv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return _bool(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def _write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _tsv_value(row.get(field, "")) for field in fields})


def _node_row(node: Node) -> dict[str, Any]:
    return asdict(node)


def _edge_row(edge: Edge) -> dict[str, Any]:
    return asdict(edge)


def _occurrence_row(occurrence: Occurrence) -> dict[str, Any]:
    return asdict(occurrence)


def _write_outputs(
    config: BuildConfig,
    layers: Sequence[Mapping[str, Any]],
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    matches: Sequence[Match],
    occurrences: Sequence[Occurrence],
    rejected: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    graph = {
        "schema_version": "pnet-graph-v1",
        "layers": list(layers),
        "nodes_file": "nodes.tsv",
        "edges_file": "edges.tsv",
        "algorithm": {
            "name": ALGORITHM_NAME,
            "version": ALGORITHM_VERSION,
            "traversal_mode": config.traversal_mode,
            "source_max_hops": config.source_max_hops,
            "target_max_hops": config.target_max_hops,
            "keyword_match": "normalized_substring_any",
            "dead_end_policy": config.dead_end_policy,
            "bridge_policy": "complete_bipartite",
        },
    }
    with (config.output_dir / "graph.yaml").open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(graph, stream, allow_unicode=True, sort_keys=False)
    _write_tsv(config.output_dir / "nodes.tsv", NODE_FIELDS, map(_node_row, nodes))
    _write_tsv(config.output_dir / "edges.tsv", EDGE_FIELDS, map(_edge_row, edges))
    _write_tsv(
        config.output_dir / "entity_matches.tsv",
        MATCH_FIELDS,
        (asdict(match) for match in matches),
    )
    _write_tsv(
        config.output_dir / "bfs_occurrences.tsv",
        OCCURRENCE_FIELDS,
        map(_occurrence_row, occurrences),
    )
    _write_tsv(config.output_dir / "rejected_edges.tsv", REJECTED_FIELDS, rejected)
    with (config.output_dir / "build_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def build_pnet(config: BuildConfig) -> BuildResult:
    """Build, validate, and write a complete PNet output directory."""
    config = config.checked()
    with _sqlite_connection(config.sqlite) as connection:
        _require_schema(connection)
        entities = _load_entities(connection)
        relations = _load_relations(connection)

    source_matches, source_seeds = _match_entities(entities, config.text_s, "source")
    target_matches, target_seeds = _match_entities(entities, config.text_t, "target")
    if not source_seeds:
        raise BuildError(f"起点关键词未匹配任何实体：{list(config.text_s)}")
    if not target_seeds:
        raise BuildError(f"终点关键词未匹配任何实体：{list(config.text_t)}")

    source = _make_side(
        "source",
        config.source_max_hops,
        source_seeds,
        source_matches,
        entities,
        relations,
        config.traversal_mode,
        config.kg_name,
    )
    target = _make_side(
        "target",
        config.target_max_hops,
        target_seeds,
        target_matches,
        entities,
        relations,
        config.traversal_mode,
        config.kg_name,
    )

    nodes = sorted(
        [*source.nodes.values(), *target.nodes.values()],
        key=lambda node: (
            node.bfs_depth
            if node.side == "source"
            else config.source_max_hops
            + 1
            + config.target_max_hops
            - node.bfs_depth,
            node.node_id,
        ),
    )
    if len(nodes) > config.max_nodes:
        raise BuildError(f"节点数 {len(nodes)} 超过 max_nodes={config.max_nodes}；构建已中止")

    bridge_count = len(source.frontier_node_ids) * len(target.frontier_node_ids)
    if bridge_count > config.max_bridge_edges:
        raise BuildError(
            f"桥边笛卡尔积 {bridge_count} 超过 max_bridge_edges={config.max_bridge_edges}；"
            "未抽样，构建已中止"
        )
    edge_map = {**source.edges, **target.edges}
    for source_node_id in source.frontier_node_ids:
        for target_node_id in target.frontier_node_ids:
            pair = (source_node_id, target_node_id)
            if pair in edge_map:
                raise BuildError(f"桥边与已有边发生 node pair 冲突：{pair}")
            edge_map[pair] = Edge(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relation_type="structural_frontier_bridge",
                knowledge_source="dual_bfs_algorithm",
                confidence=1.0,
                is_fallback=False,
                edge_kind="structural_bridge",
                is_structural=True,
                traversal_direction="structural",
            )
    edges = sorted(edge_map.values(), key=lambda edge: (edge.source_node_id, edge.target_node_id))
    if len(edges) > config.max_edges:
        raise BuildError(f"边数 {len(edges)} 超过 max_edges={config.max_edges}；构建已中止")

    layers = _layers(config.source_max_hops, config.target_max_hops)
    _validate(nodes, edges, layers, source.frontier_node_ids, target.frontier_node_ids)
    matches = sorted(
        [*source_matches, *target_matches],
        key=lambda item: (
            item.side,
            item.entity_id,
            item.keyword,
            item.matched_field,
            item.matched_text,
        ),
    )
    occurrences = sorted(
        [*source.occurrences, *target.occurrences],
        key=lambda item: (item.side, item.bfs_depth, item.node_id),
    )
    rejected = [
        *_rejected_edges(
            "source",
            config.source_max_hops,
            source.distances,
            relations,
            config.traversal_mode,
        ),
        *_rejected_edges(
            "target",
            config.target_max_hops,
            target.distances,
            relations,
            config.traversal_mode,
        ),
    ]
    rejected.sort(
        key=lambda row: (
            row["side"], row["source_entity_id"], row["target_entity_id"], row["relation_id"]
        )
    )

    layer_widths = [
        sum(node.layer == str(layer["id"]) for node in nodes) for layer in layers
    ]
    sqlite_stat = config.sqlite.stat()
    kg_version = config.kg_version or (
        f"sqlite-size-{sqlite_stat.st_size}-mtime_ns-{sqlite_stat.st_mtime_ns}"
    )
    manifest: dict[str, Any] = {
        "schema_version": "dual-bfs-pnet-manifest-v1",
        "algorithm": {
            "name": ALGORITHM_NAME,
            "version": ALGORITHM_VERSION,
            "keyword_match_rule": "normalized_substring_any",
            "normalization": "unicode_nfkc_lower_trim_whitespace_punctuation_v1",
            "traversal_mode": config.traversal_mode,
            "source_max_hops": config.source_max_hops,
            "target_max_hops": config.target_max_hops,
            "dead_end_policy": config.dead_end_policy,
            "bridge_policy": "complete_bipartite",
            "deterministic_sort": "entity_id_then_relation_id",
        },
        "inputs": {
            "kg_name": config.kg_name,
            "kg_version": kg_version,
            "kg_content_hash": _hash_file(config.sqlite),
            "text_s": list(config.text_s),
            "text_t": list(config.text_t),
        },
        "matching": {
            "source_matched_entity_count": len(source_seeds),
            "target_matched_entity_count": len(target_seeds),
            "source_match_record_count": len(source_matches),
            "target_match_record_count": len(target_matches),
        },
        "graph": {
            "source_frontier_count": len(source.frontier_node_ids),
            "target_frontier_count": len(target.frontier_node_ids),
            "expected_bridge_edge_count": bridge_count,
            "actual_bridge_edge_count": bridge_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "layer_widths": layer_widths,
            "carry_node_count": source.carry_node_count + target.carry_node_count,
            "carry_edge_count": source.carry_edge_count + target.carry_edge_count,
            "rejected_edge_record_count": len(rejected),
        },
        "limits": {
            "max_nodes": config.max_nodes,
            "max_edges": config.max_edges,
            "max_bridge_edges": config.max_bridge_edges,
        },
        "status": {
            "search_complete": True,
            "bridge_complete": True,
            "validation_passed": True,
        },
    }
    _write_outputs(config, layers, nodes, edges, matches, occurrences, rejected, manifest)
    return BuildResult(
        output_dir=config.output_dir,
        node_count=len(nodes),
        edge_count=len(edges),
        source_match_count=len(source_seeds),
        target_match_count=len(target_seeds),
        bridge_edge_count=bridge_count,
        manifest=manifest,
    )


def _config_from_args(args: argparse.Namespace) -> BuildConfig:
    values: dict[str, Any] = {}
    if args.config:
        with args.config.open(encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, dict):
            raise BuildError("配置文件顶层必须是 JSON object")
        values.update(loaded)
    overrides = {
        "sqlite": args.sqlite,
        "output_dir": args.output_dir,
        "text_s": args.text_s,
        "text_t": args.text_t,
        "source_max_hops": args.source_max_hops,
        "target_max_hops": args.target_max_hops,
        "traversal_mode": args.traversal_mode,
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
        "max_bridge_edges": args.max_bridge_edges,
        "kg_name": args.kg_name,
        "kg_version": args.kg_version,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    required = [key for key in ("sqlite", "output_dir", "text_s", "text_t") if key not in values]
    if required:
        raise BuildError(f"缺少配置项：{', '.join(required)}")
    values["sqlite"] = Path(values["sqlite"])
    values["output_dir"] = Path(values["output_dir"])
    values["text_s"] = tuple(values["text_s"])
    values["text_t"] = tuple(values["text_t"])
    try:
        return BuildConfig(**values)
    except TypeError as exc:
        raise BuildError(f"配置字段错误：{exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Gold SQLite 构建 Dual-Keyword Bidirectional BFS PNet。"
    )
    parser.add_argument("--config", type=Path, help="JSON 构建配置；命令行参数可覆盖其字段")
    parser.add_argument("--sqlite", type=Path, help="Gold SQLite 路径")
    parser.add_argument("--output-dir", type=Path, help="输出目录")
    parser.add_argument("--text-s", action="append", help="起点关键词；可重复")
    parser.add_argument("--text-t", action="append", help="终点关键词；可重复")
    parser.add_argument("--source-max-hops", type=int)
    parser.add_argument("--target-max-hops", type=int)
    parser.add_argument("--traversal-mode", choices=("undirected", "directed"))
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-edges", type=int)
    parser.add_argument("--max-bridge-edges", type=int)
    parser.add_argument("--kg-name")
    parser.add_argument("--kg-version")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        config = _config_from_args(parser.parse_args(argv))
        result = build_pnet(config)
    except (BuildError, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(f"PNet 构建失败：{exc}", file=sys.stderr)
        return 2
    print(
        "PNet 构建完成："
        f"source_matches={result.source_match_count}, "
        f"target_matches={result.target_match_count}, "
        f"nodes={result.node_count}, edges={result.edge_count}, "
        f"bridges={result.bridge_edge_count}, output={result.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
