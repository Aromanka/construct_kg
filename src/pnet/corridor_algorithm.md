# BBC-PNet 实现约定

## 算法

默认算法名为 `bounded_bidirectional_corridor_pnet`，版本为 `1.0`。

给定关键词匹配实体集合 `S`、`T` 和最大边数 `H`：

1. 从 `S` 做有界多源 BFS，得到 `ds`；
2. 从 `T` 做反向有界多源 BFS，得到 `dt`；
3. 走廊实体满足 `ds(v) + dt(v) <= H`；
4. `dt` 减少 1 的 KG 边为 `kg_progress`；
5. `dt` 相等的 KG 边为 `kg_lateral`，无向模式只保留实体 ID 字典序方向；
6. `dt` 增大的后退边不进入 PNet；
7. 以 `(entity_id, step, lateral_steps)` 为状态进行 `H+1` 层时间展开；
8. 每次扩展必须满足 `step + 1 + dt(v) <= H`；
9. 提前到达终点后只沿 `terminal_carry` 传播到最终层；
10. 从最终层反向裁剪，并再次检查从第 0 层的正向可达性。

有向模式下，`ds` 使用原始关系方向，`dt` 使用反向关系。无向模式下，关系方向仅作为
审计信息保留，PNet 计算方向由 `dt` 单调性决定。

## 规模剪枝

默认超预算策略是 `protected_backbone_degree_pruning`。实体评分固定为：

```text
log1p(corridor_degree)
+ keyword_relevance
+ 0.25 * mean_incident_relation_confidence
- 0.05 * (ds + dt)
```

同分时按 `entity_id` 排序，保证结果确定。剪枝顺序为：

```text
unique corridor entities
→ per-layer unique entities
→ occurrence nodes
→ edges
→ reachability pruning
```

`max_entities_per_layer` 按原始 KG 实体或 terminal carry 对应的终点实体计数；同一实体的
不同 lateral state 及其 carry 不重复占用该实体预算。实际 PNet 实例宽度另由
`max_occurrence_nodes` 控制，并在 `graph.layer_widths` 中报告；每层实体宽度记录在
`graph.layer_unique_entity_widths`。

无条件保护内容包括可连接起点的最短前进骨架、每个保留最终节点的完整反向路径，以及这些
路径需要的 terminal carry。`keep_all_progress_edges=true` 时，若某实体的前进边数量本身
超过 `max_edges_per_node`，前进边仍全部保留，并在 manifest 的
`progress_cap_overflow_nodes` 中报告；该上限优先用于限制横向增密边。

任何预算剪枝都会令 `status.search_complete=false`。这表示输出是确定性预算子图，而不是
完整走廊；`validation_passed=true` 仅表示该预算子图结构合法且每个节点属于完整通路。

## 输出语义

- `nodes.tsv` 同时报告原始 KG 实体和时间展开实例，并附带 `source_distance`、
  `target_distance`、`lateral_steps`。
- `edges.tsv` 的所有 `kg_*` 边均包含 Gold assertion ID 作为
  `evidence_relation_ids`。
- `terminal_carry` 是人工对齐结构，`is_structural=true`，不能解释成医学关系。
- `structural_frontier_bridge` 数量始终为 0。
- `build_manifest.json` 同时报告候选唯一实体、选中唯一实体、最终实际实体和 occurrence
  node 数量。
