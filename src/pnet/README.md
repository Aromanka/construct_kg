# Gold 图谱 PNet 构建

`build_pnet.py` 按 [requirement.md](requirement.md) 从项目的 Gold SQLite 图谱构建
Dual-Keyword Bidirectional BFS PNet。默认交付配置位于 `build_config.json`，使用：

- 起点关键词：`acoustic voice characteristics`
- 终点关键词：`type 2 diabetes mellitus`
- 起点/终点最大跳数：各 1 跳
- 遍历模式：`undirected`
- 断头分支策略：`structural_carry`

默认关键词来自当前 Gold 数据并能得到非空匹配。关键词使用 normalized substring OR
语义，因此修改为更宽泛的词（例如仅 `diabetes`）可能显著扩大 BFS 前沿及桥边笛卡尔积。

## 构建命令

在项目根目录、已激活项目 Python 环境后执行：

```powershell
python src/pnet/build_pnet.py --config src/pnet/build_config.json
```

命令行参数可以覆盖 JSON 配置。例如重新选择关键词和跳数：

```powershell
python src/pnet/build_pnet.py `
  --config src/pnet/build_config.json `
  --text-s "voice biomarkers" `
  --text-t "type 2 diabetes mellitus" `
  --source-max-hops 1 `
  --target-max-hops 1 `
  --output-dir src/pnet/pnet_output_custom
```

同一侧传入多个关键词时重复使用 `--text-s` 或 `--text-t`。命令行一旦提供某一侧关键词，
会整体替换配置文件中该侧的列表。

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

构建器在写文件前检查层非空、节点/边唯一性、相邻层约束、DAG 可达性、KG 证据、人工边
标记以及完全二部桥边数量。超过 `max_nodes`、`max_edges` 或 `max_bridge_edges` 时直接失败，
不会抽样或写出被截断的正式产物。

## 验证命令

运行 PNet 专项测试：

```powershell
pytest tests/test_pnet_builder.py -q
```

运行项目全量测试和静态检查：

```powershell
pytest -q
ruff check .
```

构建完成后可以查看 manifest 的验收状态：

```powershell
Get-Content -Raw src/pnet/pnet_output/build_manifest.json
```
