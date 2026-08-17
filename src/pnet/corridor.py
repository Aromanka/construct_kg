"""Bounded Bidirectional Corridor PNet implementation.

The implementation deliberately avoids path enumeration.  It computes bounded
multi-source distances, extracts the ``ds + dt <= H`` corridor, time-expands it,
and then prunes the resulting DAG from its terminal layer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote

import yaml

try:
    from .build_pnet import (
        MATCH_FIELDS,
        BuildConfig,
        BuildError,
        BuildResult,
        Edge,
        Entity,
        Match,
        Node,
        Occurrence,
        Relation,
        _adjacency,
        _bfs,
        _hash_file,
        _load_entities,
        _load_relations,
        _match_entities,
        _require_schema,
        _sqlite_connection,
        _write_tsv,
    )
except ImportError:  # Direct execution through build_pnet.py.
    from build_pnet import (  # type: ignore[no-redef]
        MATCH_FIELDS,
        BuildConfig,
        BuildError,
        BuildResult,
        Edge,
        Entity,
        Match,
        Node,
        Occurrence,
        Relation,
        _adjacency,
        _bfs,
        _hash_file,
        _load_entities,
        _load_relations,
        _match_entities,
        _require_schema,
        _sqlite_connection,
        _write_tsv,
    )


ALGORITHM_NAME = "bounded_bidirectional_corridor_pnet"
ALGORITHM_VERSION = "1.0"

CORRIDOR_NODE_FIELDS = (
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
    "source_distance",
    "target_distance",
    "lateral_steps",
    "terminal_entity_id",
)
CORRIDOR_EDGE_FIELDS = (
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
CORRIDOR_OCCURRENCE_FIELDS = (
    "side",
    "node_id",
    "entity_id",
    "bfs_depth",
    "is_seed",
    "is_carry",
    "parent_node_ids",
    "parent_relation_ids",
    "source_distance",
    "target_distance",
    "lateral_steps",
)
CORRIDOR_REJECTED_FIELDS = (
    "side",
    "source_entity_id",
    "target_entity_id",
    "relation_id",
    "source_depth",
    "target_depth",
    "rejection_reason",
)


@dataclass(frozen=True, order=True)
class State:
    kind: str
    entity_id: str
    step: int
    lateral_steps: int = 0


@dataclass(frozen=True)
class Arc:
    source_entity_id: str
    target_entity_id: str
    edge_kind: str
    relations: tuple[Relation, ...]
    traversals: tuple[str, ...]


@dataclass
class CorridorStats:
    candidate_unique_entities: int = 0
    retained_unique_entities: int = 0
    unique_entities_pruned: int = 0
    candidate_occurrences: int = 0
    occurrences_after_budget: int = 0
    occurrence_nodes_pruned: int = 0
    candidate_edges: int = 0
    edges_after_budget: int = 0
    edges_pruned: int = 0
    per_node_arcs_pruned: int = 0
    progress_cap_overflow_nodes: int = 0

    @property
    def budget_pruning_applied(self) -> bool:
        return any(
            (
                self.unique_entities_pruned,
                self.occurrence_nodes_pruned,
                self.edges_pruned,
                self.per_node_arcs_pruned,
            )
        )


def _state_node_id(state: State) -> str:
    encoded = quote(state.entity_id, safe="-._~")
    if state.kind == "carry":
        digest = hashlib.sha256(state.entity_id.encode()).hexdigest()[:16]
        return f"step::d{state.step:03d}::carry::{digest}"
    return (
        f"step::d{state.step:03d}::lat{state.lateral_steps:03d}::{encoded}"
    )


def _carry_entity_id(target_entity_id: str, step: int) -> str:
    digest = hashlib.sha256(target_entity_id.encode()).hexdigest()[:16]
    return f"STRUCTURAL_TERMINAL_CARRY::{digest}::d{step:03d}"


def _state_anchor(state: State) -> str:
    return state.entity_id


def _layer_id(step: int) -> str:
    return f"step_{step:03d}"


def _distance_corridor(
    source_seeds: set[str],
    target_seeds: set[str],
    relations: Sequence[Relation],
    traversal_mode: str,
    max_hops: int,
) -> tuple[dict[str, int], dict[str, int], set[str]]:
    source_distances, _ = _bfs(
        source_seeds, relations, "source", traversal_mode, max_hops
    )
    target_distances, _ = _bfs(
        target_seeds, relations, "target", traversal_mode, max_hops
    )
    corridor = {
        entity_id
        for entity_id, source_distance in source_distances.items()
        if entity_id in target_distances
        and source_distance + target_distances[entity_id] <= max_hops
    }
    return source_distances, target_distances, corridor


def _protected_backbone(
    source_seeds: set[str],
    target_seeds: set[str],
    corridor: set[str],
    target_distances: Mapping[str, int],
    relations: Sequence[Relation],
    traversal_mode: str,
) -> set[str]:
    adjacency = _adjacency(relations, "source", traversal_mode)
    protected_entities = (source_seeds | target_seeds) & corridor
    for seed in sorted(source_seeds & corridor):
        current = seed
        while current not in target_seeds:
            current_distance = target_distances[current]
            choices = [
                (neighbor, relation.relation_id)
                for neighbor, relation in adjacency.get(current, ())
                if neighbor in corridor
                and target_distances.get(neighbor) == current_distance - 1
            ]
            if not choices:
                raise BuildError(f"无法为走廊起点重建最短骨架：{seed}")
            neighbor, _relation_id = min(choices)
            protected_entities.add(neighbor)
            current = neighbor
    return protected_entities


def _entity_scores(
    corridor: set[str],
    source_seeds: set[str],
    target_seeds: set[str],
    source_distances: Mapping[str, int],
    target_distances: Mapping[str, int],
    relations: Sequence[Relation],
) -> dict[str, float]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    confidence_sum: dict[str, float] = defaultdict(float)
    confidence_count: dict[str, int] = defaultdict(int)
    for relation in relations:
        source = relation.source_entity_id
        target = relation.target_entity_id
        if source not in corridor or target not in corridor or source == target:
            continue
        neighbors[source].add(target)
        neighbors[target].add(source)
        confidence_sum[source] += relation.confidence
        confidence_sum[target] += relation.confidence
        confidence_count[source] += 1
        confidence_count[target] += 1
    scores: dict[str, float] = {}
    keyword_entities = source_seeds | target_seeds
    for entity_id in corridor:
        degree = len(neighbors.get(entity_id, ()))
        confidence = confidence_sum[entity_id] / max(1, confidence_count[entity_id])
        distance_sum = source_distances[entity_id] + target_distances[entity_id]
        keyword_relevance = 1.0 if entity_id in keyword_entities else 0.0
        scores[entity_id] = (
            math.log1p(degree)
            + keyword_relevance
            + 0.25 * confidence
            - 0.05 * distance_sum
        )
    return scores


def _select_corridor_entities(
    config: BuildConfig,
    corridor: set[str],
    protected_entities: set[str],
    scores: Mapping[str, float],
    stats: CorridorStats,
) -> set[str]:
    stats.candidate_unique_entities = len(corridor)
    if len(corridor) <= config.max_unique_entities:
        stats.retained_unique_entities = len(corridor)
        return set(corridor)
    if config.overflow_policy == "fail":
        raise BuildError(
            f"走廊实体数 {len(corridor)} 超过 "
            f"max_unique_entities={config.max_unique_entities}"
        )
    if len(protected_entities) > config.max_unique_entities:
        raise BuildError(
            f"受保护骨架已有 {len(protected_entities)} 个实体，超过 "
            f"max_unique_entities={config.max_unique_entities}；不能安全截断"
        )
    remaining = sorted(
        corridor - protected_entities,
        key=lambda entity_id: (-scores[entity_id], entity_id),
    )
    capacity = config.max_unique_entities - len(protected_entities)
    selected = protected_entities | set(remaining[:capacity])
    stats.retained_unique_entities = len(selected)
    stats.unique_entities_pruned = len(corridor) - len(selected)
    return selected


def _relation_orientations(
    relation: Relation,
    traversal_mode: str,
) -> Iterable[tuple[str, str, str]]:
    source = relation.source_entity_id
    target = relation.target_entity_id
    yield source, target, "forward"
    if traversal_mode == "undirected" and source != target:
        yield target, source, "reverse"


def _build_arcs(
    config: BuildConfig,
    selected_entities: set[str],
    target_distances: Mapping[str, int],
    relations: Sequence[Relation],
    stats: CorridorStats,
) -> tuple[dict[str, list[Arc]], set[tuple[str, str]]]:
    grouped: dict[tuple[str, str, str], list[tuple[Relation, str]]] = defaultdict(list)
    rejected_by_cap: set[tuple[str, str]] = set()
    for relation in relations:
        for source, target, traversal in _relation_orientations(
            relation, config.traversal_mode
        ):
            if source not in selected_entities or target not in selected_entities:
                continue
            source_distance = target_distances[source]
            target_distance = target_distances[target]
            edge_kind = ""
            if target_distance == source_distance - 1:
                edge_kind = "kg_progress"
            elif (
                config.allow_lateral_edges
                and target_distance == source_distance
                and (
                    config.traversal_mode == "directed"
                    or source < target
                )
            ):
                edge_kind = "kg_lateral"
            if edge_kind:
                grouped[(source, target, edge_kind)].append((relation, traversal))

    arcs_by_source: dict[str, list[Arc]] = defaultdict(list)
    for (source, target, edge_kind), evidence in sorted(grouped.items()):
        arcs_by_source[source].append(
            Arc(
                source_entity_id=source,
                target_entity_id=target,
                edge_kind=edge_kind,
                relations=tuple(item[0] for item in evidence),
                traversals=tuple(sorted({item[1] for item in evidence})),
            )
        )

    for source in list(arcs_by_source):
        arcs = sorted(
            arcs_by_source[source],
            key=lambda arc: (
                0 if arc.edge_kind == "kg_progress" else 1,
                arc.target_entity_id,
                tuple(relation.relation_id for relation in arc.relations),
            ),
        )
        progress = [arc for arc in arcs if arc.edge_kind == "kg_progress"]
        lateral = [arc for arc in arcs if arc.edge_kind == "kg_lateral"]
        if config.keep_all_progress_edges:
            lateral_capacity = max(0, config.max_edges_per_node - len(progress))
            kept = [*progress, *lateral[:lateral_capacity]]
            if len(progress) > config.max_edges_per_node:
                stats.progress_cap_overflow_nodes += 1
        else:
            kept = arcs[: config.max_edges_per_node]
        for arc in arcs:
            if arc not in kept:
                rejected_by_cap.add((arc.source_entity_id, arc.target_entity_id))
        stats.per_node_arcs_pruned += len(arcs) - len(kept)
        arcs_by_source[source] = kept
    return arcs_by_source, rejected_by_cap


def _edge_from_arc(source: State, target: State, arc: Arc, kg_name: str) -> Edge:
    relations = arc.relations
    return Edge(
        source_node_id=_state_node_id(source),
        target_node_id=_state_node_id(target),
        relation_type="|".join(sorted({relation.relation_type for relation in relations})),
        knowledge_source=kg_name,
        confidence=max(relation.confidence for relation in relations),
        is_fallback=False,
        evidence_relation_ids="|".join(
            sorted({relation.relation_id for relation in relations})
        ),
        edge_kind=arc.edge_kind,
        is_structural=False,
        original_source_entity_id="|".join(
            sorted({relation.source_entity_id for relation in relations})
        ),
        original_target_entity_id="|".join(
            sorted({relation.target_entity_id for relation in relations})
        ),
        original_direction="|".join(
            sorted(
                {
                    f"{relation.source_entity_id}->{relation.target_entity_id}"
                    for relation in relations
                }
            )
        ),
        traversal_direction="|".join(arc.traversals),
    )


def _carry_edge(source: State, target: State) -> Edge:
    return Edge(
        source_node_id=_state_node_id(source),
        target_node_id=_state_node_id(target),
        relation_type="structural_terminal_carry",
        knowledge_source="dual_bfs_algorithm",
        confidence=1.0,
        is_fallback=True,
        edge_kind="terminal_carry",
        is_structural=True,
        traversal_direction="structural",
    )


def _time_expand(
    config: BuildConfig,
    active_sources: set[str],
    target_seeds: set[str],
    selected_entities: set[str],
    target_distances: Mapping[str, int],
    arcs_by_source: Mapping[str, Sequence[Arc]],
) -> tuple[set[State], dict[tuple[State, State], Edge]]:
    states_by_step: list[set[State]] = [set() for _ in range(config.max_layers)]
    states_by_step[0] = {
        State("entity", entity_id, 0, 0)
        for entity_id in active_sources & selected_entities
    }
    edges: dict[tuple[State, State], Edge] = {}
    for step in range(config.max_hops):
        for state in sorted(states_by_step[step]):
            if state.kind == "carry":
                next_state = State("carry", state.entity_id, step + 1, 0)
                states_by_step[step + 1].add(next_state)
                edges[(state, next_state)] = _carry_edge(state, next_state)
                continue
            if state.entity_id in target_seeds:
                next_state = State("carry", state.entity_id, step + 1, 0)
                states_by_step[step + 1].add(next_state)
                edges[(state, next_state)] = _carry_edge(state, next_state)
                continue
            for arc in arcs_by_source.get(state.entity_id, ()):
                target_entity = arc.target_entity_id
                next_lateral = state.lateral_steps + (
                    1 if arc.edge_kind == "kg_lateral" else 0
                )
                if next_lateral > config.max_lateral_steps:
                    continue
                if step + 1 + target_distances[target_entity] > config.max_hops:
                    continue
                next_state = State("entity", target_entity, step + 1, next_lateral)
                states_by_step[step + 1].add(next_state)
                edges[(state, next_state)] = _edge_from_arc(
                    state, next_state, arc, config.kg_name
                )
    return set().union(*states_by_step), edges


def _graph_indexes(
    edges: Mapping[tuple[State, State], Edge],
) -> tuple[dict[State, list[State]], dict[State, list[State]]]:
    outgoing: dict[State, list[State]] = defaultdict(list)
    incoming: dict[State, list[State]] = defaultdict(list)
    for source, target in edges:
        outgoing[source].append(target)
        incoming[target].append(source)
    for values in outgoing.values():
        values.sort(key=_state_node_id)
    for values in incoming.values():
        values.sort(key=_state_node_id)
    return outgoing, incoming


def _complete_path_states(
    states: set[State],
    edges: Mapping[tuple[State, State], Edge],
    target_seeds: set[str],
    max_hops: int,
) -> tuple[set[State], dict[tuple[State, State], Edge]]:
    outgoing, incoming = _graph_indexes(edges)
    terminal = {
        state
        for state in states
        if state.step == max_hops
        and (state.kind == "carry" or state.entity_id in target_seeds)
    }
    reverse_reachable = set(terminal)
    queue = deque(sorted(terminal))
    while queue:
        state = queue.popleft()
        for parent in incoming.get(state, ()):
            if parent not in reverse_reachable:
                reverse_reachable.add(parent)
                queue.append(parent)
    starts = {state for state in reverse_reachable if state.step == 0}
    forward_reachable = set(starts)
    queue = deque(sorted(starts))
    while queue:
        state = queue.popleft()
        for child in outgoing.get(state, ()):
            if child in reverse_reachable and child not in forward_reachable:
                forward_reachable.add(child)
                queue.append(child)
    kept_states = reverse_reachable & forward_reachable
    kept_edges = {
        pair: edge
        for pair, edge in edges.items()
        if pair[0] in kept_states and pair[1] in kept_states
    }
    return kept_states, kept_edges


def _protect_occurrence_paths(
    states: set[State],
    edges: Mapping[tuple[State, State], Edge],
    max_hops: int,
) -> tuple[set[State], set[tuple[State, State]]]:
    outgoing, incoming = _graph_indexes(edges)
    protected_states: set[State] = set()
    protected_edges: set[tuple[State, State]] = set()

    def protect_forward(start: State) -> None:
        current = start
        protected_states.add(current)
        while current.step < max_hops:
            choices = outgoing.get(current, ())
            if not choices:
                raise BuildError(f"受保护起点没有完整终点通路：{_state_node_id(start)}")
            child = min(
                choices,
                key=lambda state: (
                    0
                    if edges[(current, state)].edge_kind == "kg_progress"
                    else 1
                    if edges[(current, state)].edge_kind == "kg_lateral"
                    else 2,
                    _state_node_id(state),
                ),
            )
            protected_edges.add((current, child))
            protected_states.add(child)
            current = child

    for start in sorted(state for state in states if state.step == 0):
        protect_forward(start)

    for terminal in sorted(state for state in states if state.step == max_hops):
        current = terminal
        protected_states.add(current)
        while current.step > 0:
            choices = incoming.get(current, ())
            if not choices:
                raise BuildError(
                    f"最终层节点没有起点通路：{_state_node_id(terminal)}"
                )
            parent = min(choices, key=_state_node_id)
            protected_edges.add((parent, current))
            protected_states.add(parent)
            current = parent
    return protected_states, protected_edges


def _state_rank(
    state: State,
    protected_states: set[State],
    entity_scores: Mapping[str, float],
) -> tuple[Any, ...]:
    return (
        0 if state in protected_states else 1,
        0 if state.kind == "carry" else 1,
        -entity_scores.get(_state_anchor(state), 0.0),
        state.lateral_steps,
        _state_node_id(state),
    )


def _prune_selected_graph(
    states: set[State],
    edges: dict[tuple[State, State], Edge],
    terminal_entities: set[str],
    max_hops: int,
) -> tuple[set[State], dict[tuple[State, State], Edge]]:
    return _complete_path_states(states, edges, terminal_entities, max_hops)


def _apply_edge_budget(
    config: BuildConfig,
    states: set[State],
    edges: dict[tuple[State, State], Edge],
    protected_edges: set[tuple[State, State]],
    terminal_entities: set[str],
    stats: CorridorStats,
) -> tuple[set[State], dict[tuple[State, State], Edge]]:
    stats.candidate_edges = len(edges)
    if len(edges) <= config.max_edges:
        stats.edges_after_budget = len(edges)
        return states, edges
    if config.overflow_policy == "fail":
        raise BuildError(f"边数 {len(edges)} 超过 max_edges={config.max_edges}")
    protected_edges &= set(edges)
    if len(protected_edges) > config.max_edges:
        raise BuildError(
            f"受保护骨架边数 {len(protected_edges)} 超过 max_edges={config.max_edges}"
        )
    remaining = sorted(
        set(edges) - protected_edges,
        key=lambda pair: (
            0 if edges[pair].edge_kind == "terminal_carry" else 1,
            0 if edges[pair].edge_kind == "kg_progress" else 1,
            _state_node_id(pair[0]),
            _state_node_id(pair[1]),
        ),
    )
    capacity = config.max_edges - len(protected_edges)
    selected_pairs = protected_edges | set(remaining[:capacity])
    selected_edges = {pair: edges[pair] for pair in selected_pairs}
    states, selected_edges = _prune_selected_graph(
        states, selected_edges, terminal_entities, config.max_hops
    )
    stats.edges_after_budget = len(selected_edges)
    stats.edges_pruned = stats.candidate_edges - len(selected_edges)
    return states, selected_edges


def _validate_corridor(
    config: BuildConfig,
    states: set[State],
    edges: Mapping[tuple[State, State], Edge],
    source_seeds: set[str],
    target_seeds: set[str],
    target_distances: Mapping[str, int],
) -> None:
    errors: list[str] = []
    node_ids = {_state_node_id(state) for state in states}
    if len(node_ids) != len(states):
        errors.append("node_id 非全局唯一")
    widths = [
        sum(state.step == step for state in states)
        for step in range(config.max_layers)
    ]
    if any(width == 0 for width in widths):
        errors.append(f"存在空层：{widths}")
    if any(
        state.kind != "entity" or state.entity_id not in source_seeds
        for state in states
        if state.step == 0
    ):
        errors.append("第 0 层包含非起点节点")
    if any(
        state.kind != "carry" and state.entity_id not in target_seeds
        for state in states
        if state.step == config.max_hops
    ):
        errors.append("最终层包含非终点实体")
    for (source, target), edge in edges.items():
        if source not in states or target not in states:
            errors.append("边引用被裁剪节点")
            continue
        if target.step != source.step + 1:
            errors.append(f"非相邻层边：{edge.source_node_id} -> {edge.target_node_id}")
        if edge.edge_kind.startswith("kg_") and not edge.evidence_relation_ids:
            errors.append(f"KG 边缺少 evidence：{edge.source_node_id}")
        if edge.edge_kind == "kg_progress":
            if target_distances[target.entity_id] != target_distances[source.entity_id] - 1:
                errors.append("progress edge 不满足 dt 单调递减")
            if target.lateral_steps != source.lateral_steps:
                errors.append("progress edge 错误改变 lateral_steps")
        elif edge.edge_kind == "kg_lateral":
            if target_distances[target.entity_id] != target_distances[source.entity_id]:
                errors.append("lateral edge 的 dt 不相等")
            if target.lateral_steps != source.lateral_steps + 1:
                errors.append("lateral edge 未增加 lateral_steps")
        elif edge.edge_kind == "terminal_carry":
            if not edge.is_structural or edge.knowledge_source != "dual_bfs_algorithm":
                errors.append("terminal carry 标记不完整")
        if (
            target.kind == "entity"
            and target.step + target_distances[target.entity_id] > config.max_hops
        ):
            errors.append("时间展开节点超过剩余跳数预算")
    terminal_entities = target_seeds
    complete_states, complete_edges = _complete_path_states(
        states, edges, terminal_entities, config.max_hops
    )
    if complete_states != states or set(complete_edges) != set(edges):
        errors.append("存在不属于完整起点—终点通路的节点或边")
    if errors:
        raise BuildError("BBC-PNet 验证失败：\n- " + "\n- ".join(errors[:20]))


def _make_output_rows(
    config: BuildConfig,
    states: set[State],
    edges: Mapping[tuple[State, State], Edge],
    entities: Mapping[str, Entity],
    source_matches: Sequence[Match],
    target_matches: Sequence[Match],
    target_seeds: set[str],
    source_distances: Mapping[str, int],
    target_distances: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keywords: dict[str, set[str]] = defaultdict(set)
    for match in (*source_matches, *target_matches):
        keywords[match.entity_id].add(match.keyword)
    _outgoing, incoming = _graph_indexes(edges)
    node_rows: list[dict[str, Any]] = []
    occurrence_rows: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda item: (item.step, _state_node_id(item))):
        node_id = _state_node_id(state)
        parents = incoming.get(state, ())
        if state.kind == "carry":
            entity = entities[state.entity_id]
            node = Node(
                node_id=node_id,
                entity_id=_carry_entity_id(state.entity_id, state.step),
                layer=_layer_id(state.step),
                node_type="structural_carry",
                display_name=f"Terminal carry: {entity.display_name}",
                is_fallback=True,
                source="dual_bfs_algorithm",
                original_node_id="|".join(_state_node_id(parent) for parent in parents),
                description="Terminal absorption carry; not medical knowledge.",
                side="corridor",
                bfs_depth=state.step,
                is_structural=True,
            )
            source_distance: Any = ""
            target_distance: Any = 0
            lateral_steps: Any = ""
            terminal_entity_id = state.entity_id
        else:
            entity = entities[state.entity_id]
            node = Node(
                node_id=node_id,
                entity_id=state.entity_id,
                layer=_layer_id(state.step),
                node_type=entity.entity_type,
                display_name=entity.display_name or entity.entity_id,
                is_fallback=False,
                source=config.kg_name,
                external_id=entity.external_id,
                description=entity.description,
                side="corridor",
                bfs_depth=state.step,
                matched_keywords="|".join(sorted(keywords.get(state.entity_id, ()))),
                is_structural=False,
            )
            source_distance = source_distances.get(state.entity_id, "")
            target_distance = target_distances[state.entity_id]
            lateral_steps = state.lateral_steps
            terminal_entity_id = state.entity_id if state.entity_id in target_seeds else ""
        row = asdict(node)
        row.update(
            {
                "source_distance": source_distance,
                "target_distance": target_distance,
                "lateral_steps": lateral_steps,
                "terminal_entity_id": terminal_entity_id,
            }
        )
        node_rows.append(row)
        parent_relations = sorted(
            {
                relation_id
                for parent in parents
                for relation_id in edges[(parent, state)].evidence_relation_ids.split("|")
                if relation_id
            }
        )
        occurrence = Occurrence(
            side="corridor",
            node_id=node_id,
            entity_id=node.entity_id,
            bfs_depth=state.step,
            is_seed=state.step == 0,
            is_carry=state.kind == "carry",
            parent_node_ids="|".join(_state_node_id(parent) for parent in parents),
            parent_relation_ids="|".join(parent_relations),
        )
        occurrence_row = asdict(occurrence)
        occurrence_row.update(
            {
                "source_distance": source_distance,
                "target_distance": target_distance,
                "lateral_steps": lateral_steps,
            }
        )
        occurrence_rows.append(occurrence_row)
    return node_rows, occurrence_rows


def _rejected_rows(
    config: BuildConfig,
    corridor: set[str],
    selected_entities: set[str],
    source_distances: Mapping[str, int],
    target_distances: Mapping[str, int],
    relations: Sequence[Relation],
    rejected_by_cap: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation in relations:
        source = relation.source_entity_id
        target = relation.target_entity_id
        reason = ""
        if source not in corridor or target not in corridor:
            reason = "outside_bounded_corridor"
        elif source not in selected_entities or target not in selected_entities:
            reason = "entity_budget_pruned"
        elif (source, target) in rejected_by_cap or (target, source) in rejected_by_cap:
            reason = "edge_per_node_pruned"
        elif config.traversal_mode == "directed":
            if target_distances[target] > target_distances[source]:
                reason = "backward"
            elif (
                target_distances[target] == target_distances[source]
                and not config.allow_lateral_edges
            ):
                reason = "lateral_disabled"
        elif (
            target_distances[target] == target_distances[source]
            and not config.allow_lateral_edges
        ):
            reason = "lateral_disabled"
        if reason:
            rows.append(
                {
                    "side": "corridor",
                    "source_entity_id": source,
                    "target_entity_id": target,
                    "relation_id": relation.relation_id,
                    "source_depth": source_distances.get(source, ""),
                    "target_depth": target_distances.get(target, ""),
                    "rejection_reason": reason,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["source_entity_id"], row["target_entity_id"], row["relation_id"]
        ),
    )


def _write_outputs(
    config: BuildConfig,
    node_rows: Sequence[Mapping[str, Any]],
    edges: Sequence[Edge],
    matches: Sequence[Match],
    occurrence_rows: Sequence[Mapping[str, Any]],
    rejected_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    graph = {
        "schema_version": "pnet-graph-v1",
        "layers": [
            {"id": _layer_id(step), "order": step}
            for step in range(config.max_layers)
        ],
        "nodes_file": "nodes.tsv",
        "edges_file": "edges.tsv",
        "algorithm": {
            "name": ALGORITHM_NAME,
            "version": ALGORITHM_VERSION,
            "max_layers": config.max_layers,
            "max_hops": config.max_hops,
            "corridor_rule": "ds_plus_dt_le_h",
            "layering": "time_expanded_with_lateral_state",
            "traversal_mode": config.traversal_mode,
            "max_lateral_steps": config.max_lateral_steps,
            "terminal_policy": config.terminal_policy,
            "overflow_policy": config.overflow_policy,
            "full_frontier_connection": False,
        },
    }
    with (config.output_dir / "graph.yaml").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        yaml.safe_dump(graph, stream, allow_unicode=True, sort_keys=False)
    _write_tsv(config.output_dir / "nodes.tsv", CORRIDOR_NODE_FIELDS, node_rows)
    _write_tsv(
        config.output_dir / "edges.tsv",
        CORRIDOR_EDGE_FIELDS,
        (asdict(edge) for edge in edges),
    )
    _write_tsv(
        config.output_dir / "entity_matches.tsv",
        MATCH_FIELDS,
        (asdict(match) for match in matches),
    )
    _write_tsv(
        config.output_dir / "bfs_occurrences.tsv",
        CORRIDOR_OCCURRENCE_FIELDS,
        occurrence_rows,
    )
    _write_tsv(
        config.output_dir / "rejected_edges.tsv",
        CORRIDOR_REJECTED_FIELDS,
        rejected_rows,
    )
    with (config.output_dir / "build_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def build_corridor_pnet(config: BuildConfig) -> BuildResult:
    """Build the default bounded bidirectional corridor PNet."""
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

    source_distances, target_distances, corridor = _distance_corridor(
        source_seeds,
        target_seeds,
        relations,
        config.traversal_mode,
        config.max_hops,
    )
    active_sources = source_seeds & corridor
    corridor_targets = target_seeds & corridor
    if not active_sources or not corridor_targets:
        raise BuildError(
            f"Gold KG 中不存在不超过 {config.max_hops} 跳的起点—终点通路；"
            "请扩大实体匹配或小幅增加 max_hops"
        )

    stats = CorridorStats()
    protected_entities = _protected_backbone(
        active_sources,
        corridor_targets,
        corridor,
        target_distances,
        relations,
        config.traversal_mode,
    )
    entity_scores = _entity_scores(
        corridor,
        source_seeds,
        target_seeds,
        source_distances,
        target_distances,
        relations,
    )
    selected_entities = _select_corridor_entities(
        config, corridor, protected_entities, entity_scores, stats
    )
    arcs_by_source, rejected_by_cap = _build_arcs(
        config, selected_entities, target_distances, relations, stats
    )
    states, edge_map = _time_expand(
        config,
        active_sources,
        corridor_targets,
        selected_entities,
        target_distances,
        arcs_by_source,
    )
    states, edge_map = _complete_path_states(
        states, edge_map, corridor_targets, config.max_hops
    )
    if not states:
        raise BuildError("走廊时间展开后没有完整起点—终点通路")
    protected_states, protected_edges = _protect_occurrence_paths(
        states, edge_map, config.max_hops
    )
    stats.candidate_occurrences = len(states)

    # Apply the layer/entity and global occurrence caps while retaining every
    # selected source/final endpoint backbone.
    if config.overflow_policy == "fail":
        layer_counts = [
            len({_state_anchor(state) for state in states if state.step == step})
            for step in range(config.max_layers)
        ]
        if len(states) > config.max_occurrence_nodes or any(
            count > config.max_entities_per_layer for count in layer_counts
        ):
            raise BuildError("时间展开图超过 occurrence/layer 预算")
        stats.occurrences_after_budget = len(states)
    else:
        selected_states: set[State] = set()
        for step in range(config.max_layers):
            layer = {state for state in states if state.step == step}
            mandatory = {state for state in protected_states if state.step == step}
            mandatory_anchors = {_state_anchor(state) for state in mandatory}
            if len(mandatory_anchors) > config.max_entities_per_layer:
                raise BuildError(
                    f"step_{step:03d} 受保护实体超过 max_entities_per_layer"
                )
            groups: dict[str, set[State]] = defaultdict(set)
            for state in layer:
                groups[_state_anchor(state)].add(state)
            optional_anchors = sorted(
                set(groups) - mandatory_anchors,
                key=lambda item: (-entity_scores.get(item, 0.0), item),
            )
            kept_anchors = mandatory_anchors | set(
                optional_anchors[
                    : config.max_entities_per_layer - len(mandatory_anchors)
                ]
            )
            selected_states.update(
                state for anchor in kept_anchors for state in groups[anchor]
            )
        if len(selected_states) > config.max_occurrence_nodes:
            if len(protected_states) > config.max_occurrence_nodes:
                raise BuildError("受保护实例超过 max_occurrence_nodes，不能安全截断")
            optional_states = sorted(
                selected_states - protected_states,
                key=lambda state: _state_rank(
                    state, protected_states, entity_scores
                ),
            )
            selected_states = protected_states | set(
                optional_states[
                    : config.max_occurrence_nodes - len(protected_states)
                ]
            )
        edge_map = {
            pair: edge
            for pair, edge in edge_map.items()
            if pair[0] in selected_states and pair[1] in selected_states
        }
        states, edge_map = _prune_selected_graph(
            selected_states, edge_map, corridor_targets, config.max_hops
        )
        stats.occurrences_after_budget = len(states)
        stats.occurrence_nodes_pruned = stats.candidate_occurrences - len(states)

    states, edge_map = _apply_edge_budget(
        config,
        states,
        edge_map,
        protected_edges,
        corridor_targets,
        stats,
    )
    _validate_corridor(
        config,
        states,
        edge_map,
        source_seeds,
        corridor_targets,
        target_distances,
    )

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
    node_rows, occurrence_rows = _make_output_rows(
        config,
        states,
        edge_map,
        entities,
        source_matches,
        target_matches,
        target_seeds,
        source_distances,
        target_distances,
    )
    edges = sorted(
        edge_map.values(),
        key=lambda edge: (edge.source_node_id, edge.target_node_id),
    )
    rejected = _rejected_rows(
        config,
        corridor,
        selected_entities,
        source_distances,
        target_distances,
        relations,
        rejected_by_cap,
    )
    final_states = {state for state in states if state.step == config.max_hops}
    retained_sources = {state.entity_id for state in states if state.step == 0}
    retained_targets = {state.entity_id for state in final_states}
    real_entities = {state.entity_id for state in states if state.kind == "entity"}
    layer_widths = [
        sum(state.step == step for state in states)
        for step in range(config.max_layers)
    ]
    layer_unique_entity_widths = [
        len({state.entity_id for state in states if state.step == step})
        for step in range(config.max_layers)
    ]
    sqlite_stat = config.sqlite.stat()
    kg_version = config.kg_version or (
        f"sqlite-size-{sqlite_stat.st_size}-mtime_ns-{sqlite_stat.st_mtime_ns}"
    )
    manifest: dict[str, Any] = {
        "schema_version": "bbc-pnet-manifest-v1",
        "algorithm": {
            "name": ALGORITHM_NAME,
            "version": ALGORITHM_VERSION,
            "max_layers": config.max_layers,
            "max_hops": config.max_hops,
            "corridor_rule": "ds_plus_dt_le_h",
            "layering": "time_expanded_with_lateral_state",
            "traversal_mode": config.traversal_mode,
            "keep_all_progress_edges": config.keep_all_progress_edges,
            "allow_lateral_edges": config.allow_lateral_edges,
            "max_lateral_steps": config.max_lateral_steps,
            "allow_backward_edges": config.allow_backward_edges,
            "terminal_policy": config.terminal_policy,
            "overflow_policy": config.overflow_policy,
            "full_frontier_connection": False,
            "deterministic_sort": "score_then_entity_id",
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
        "coverage": {
            "source_entities_with_bounded_path": len(active_sources),
            "target_entities_in_corridor": len(corridor_targets),
            "retained_source_entity_count": len(retained_sources),
            "retained_terminal_entity_count": len(retained_targets),
        },
        "corridor": {
            "candidate_unique_entity_count": stats.candidate_unique_entities,
            "selected_unique_entity_count": stats.retained_unique_entities,
            "output_unique_real_entity_count": len(real_entities),
            "unique_entities_pruned": stats.unique_entities_pruned,
        },
        "graph": {
            "occurrence_node_count": len(states),
            "edge_count": len(edges),
            "layer_widths": layer_widths,
            "layer_unique_entity_widths": layer_unique_entity_widths,
            "progress_edge_count": sum(
                edge.edge_kind == "kg_progress" for edge in edges
            ),
            "lateral_edge_count": sum(
                edge.edge_kind == "kg_lateral" for edge in edges
            ),
            "terminal_carry_edge_count": sum(
                edge.edge_kind == "terminal_carry" for edge in edges
            ),
            "structural_bridge_edge_count": 0,
            "rejected_edge_record_count": len(rejected),
        },
        "pruning": {
            "applied": stats.budget_pruning_applied,
            "candidate_occurrence_node_count": stats.candidate_occurrences,
            "occurrence_nodes_pruned": stats.occurrence_nodes_pruned,
            "candidate_edge_count": stats.candidate_edges,
            "edges_pruned": stats.edges_pruned,
            "per_node_arcs_pruned": stats.per_node_arcs_pruned,
            "progress_cap_overflow_nodes": stats.progress_cap_overflow_nodes,
            "score": "log1p(corridor_degree)+keyword+0.25*confidence-0.05*(ds+dt)",
        },
        "limits": {
            "max_unique_entities": config.max_unique_entities,
            "max_occurrence_nodes": config.max_occurrence_nodes,
            "max_edges": config.max_edges,
            "max_entities_per_layer": config.max_entities_per_layer,
            "max_edges_per_node": config.max_edges_per_node,
        },
        "status": {
            "distance_search_complete": True,
            "corridor_extraction_complete": True,
            "search_complete": not stats.budget_pruning_applied,
            "bridge_complete": True,
            "validation_passed": True,
        },
    }
    _write_outputs(
        config,
        node_rows,
        edges,
        matches,
        occurrence_rows,
        rejected,
        manifest,
    )
    return BuildResult(
        output_dir=config.output_dir,
        node_count=len(states),
        edge_count=len(edges),
        source_match_count=len(source_seeds),
        target_match_count=len(target_seeds),
        bridge_edge_count=0,
        manifest=manifest,
    )
