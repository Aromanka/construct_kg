# Gold 图谱 PNet 构建

默认构建算法是 **Bounded Bidirectional Corridor PNet（BBC-PNet）**。旧版
Dual-Keyword Bidirectional BFS Frontier Bridge 算法仍然保留，可通过配置显式启用。

BBC-PNet 不枚举全部简单路径，也不创建前沿完全二部桥。它计算起点距离 `ds` 和终点距离
`dt`，保留满足 `ds + dt <= max_hops` 的真实 KG 短通路走廊，再进行固定深度时间展开。
完整设计和输出语义见 [corridor_algorithm.md](corridor_algorithm.md)。

## 默认构建命令

在项目根目录、已激活项目 Python 环境后执行：

```powershell
python src/pnet/build_pnet.py --config src/pnet/build_config.json
```

默认配置使用 8 层、最多 7 条边，允许每条时间展开路径最多 1 条横向 KG 边：

```json
{
  "algorithm": "bounded_bidirectional_corridor_pnet",
  "max_layers": 8,
  "max_hops": 7,
  "traversal_mode": "undirected",
  "allow_lateral_edges": true,
  "max_lateral_steps": 1,
  "allow_backward_edges": false,
  "terminal_policy": "structural_carry",
  "overflow_policy": "protected_backbone_degree_pruning"
}
```

命令行可以覆盖主要规模参数：

```powershell
python src/pnet/build_pnet.py `
  --config src/pnet/build_config.json `
  --max-layers 9 `
  --max-hops 8 `
  --max-unique-entities 8000 `
  --max-occurrence-nodes 30000 `
  --max-entities-per-layer 3000 `
  --max-edges 300000
```

`max_layers` 必须等于 `max_hops + 1`。同一侧传入多个临时关键词时重复使用
`--text-s` 或 `--text-t`；命令行提供的列表会整体替换 JSON 中对应列表。

## 超预算行为

默认 `overflow_policy=protected_backbone_degree_pruning`，不会因为普通候选节点超过预算便
立即终止。构建器会：

1. 保护每个可连接起点到终点的确定性最短 KG 骨架；
2. 保护最终层终点的反向完整通路；
3. 按走廊内部度、关键词相关性、关系置信度和 `ds+dt` 排名补充节点；
4. 执行时间层预算、实例预算和边预算；
5. 再次进行正向/反向可达裁剪。

若受保护骨架本身已经超过某项预算，构建仍会失败，因为继续截断会破坏必须保留的完整
通路。若希望任何超预算都失败，可设置：

```json
{
  "overflow_policy": "fail"
}
```

发生预算剪枝时，产物仍满足层级、证据和可达性校验，但 manifest 会明确记录：

```json
{
  "pruning": {"applied": true},
  "status": {
    "search_complete": false,
    "validation_passed": true
  }
}
```

## 使用旧版前沿桥接算法

旧算法保留在同一入口中。显式指定算法即可运行：

```powershell
python src/pnet/build_pnet.py `
  --config src/pnet/build_config.json `
  --algorithm dual_keyword_bfs_frontier_bridge `
  --source-max-hops 2 `
  --target-max-hops 2 `
  --max-nodes 200000 `
  --max-edges 4000000 `
  --max-bridge-edges 2000000 `
  --output-dir src/pnet/pnet_output_legacy
```

旧算法仍严格执行前沿完全连接，超过 `max_nodes`、`max_edges` 或
`max_bridge_edges` 时直接失败，不进行静默抽样。

## 输出

默认输出目录为 `src/pnet/pnet_output/`：

```text
graph.yaml
nodes.tsv
edges.tsv
entity_matches.tsv
bfs_occurrences.tsv
rejected_edges.tsv
build_manifest.json
```

BBC-PNet 的 `node_id` 包含时间层和横向步数，例如：

```text
step::d003::lat001::ENT_...
```

这使 `max_lateral_steps` 成为严格的逐路径约束，而不会因同一实体同层状态合并而产生超过
横向预算的隐含路径。`edges.tsv` 中主要边类型为 `kg_progress`、`kg_lateral` 和
`terminal_carry`；BBC-PNet 不生成 `structural_frontier_bridge`。

## 验证命令

```powershell
pytest tests/test_pnet_builder.py -q
pytest -q
ruff check .
```

构建完成后查看审计清单：

```powershell
Get-Content -Raw src/pnet/pnet_output/build_manifest.json
```
