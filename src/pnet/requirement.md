## 第三种 PNet：双端关键词 BFS + 前沿全连接

建议命名为：

```text
Dual-Keyword Bidirectional BFS PNet
双端关键词 BFS 前沿桥接 PNet
```

它与已有两种方案的区别是：不依赖调和中心性，也不枚举起点到终点之间的全部简单路径，而是分别从两组关键词独立扩展局部知识结构，再用人工结构边连接两个 BFS 的最外层。

---

## 1. 输入定义

### 1.1 知识图谱

知识图谱记为：

\[
G=(V,E)
\]

实体至少应包含：

```text
entity_id
display_name
aliases（可选）
entity_type（可选）
description（可选）
```

关系至少应包含：

```text
relation_id
source_entity_id
target_entity_id
relation_type
knowledge_source
confidence（可选）
```

### 1.2 关键词

输入两个非空关键词列表：

```text
text_s：起点关键词列表
text_t：重点/终点关键词列表
```

例如：

```json
{
  "text_s": [
    "heart sound",
    "cardiac auscultation"
  ],
  "text_t": [
    "diabetes",
    "diabetes mellitus"
  ]
}
```

注意：这里是两个“关键词列表”，不是只能各有一个关键词。

### 1.3 BFS 参数

建议至少提供：

```text
source_max_hops：起点侧 BFS 最大跳数 Hs
target_max_hops：终点侧 BFS 最大跳数 Ht
traversal_mode：undirected / directed
```

默认推荐：

```text
traversal_mode = undirected
```

原因是 OptimusKG 等医学知识图谱的关系方向通常不统一。此处的 PNet 传播方向只是网络计算方向，不能解释为医学因果方向。

如果外部项目已经筛选出方向语义统一的关系，也可以使用：

```text
起点侧：沿 outbound 关系搜索
终点侧：沿 inbound 关系搜索
```

---

## 2. 关键词检索

### 2.1 文本规范化

关键词和实体文本应采用相同的规范化流程：

1. Unicode NFKC 规范化；
2. 转为小写；
3. 去除首尾空格；
4. 连续空白合并为一个空格；
5. 可选：统一连字符、下划线和常见标点。

实体的检索文本建议由以下字段组成：

```text
display_name + aliases + description
```

不建议默认对 `entity_id` 做包含匹配，除非关键词本身就是标准实体 ID。

### 2.2 匹配规则

对任意实体 \(v\)，只要其检索文本包含列表中的任意一个关键词，即认为匹配成功：

\[
match(v,T)=\exists t\in T,\ normalize(t)\subseteq normalize(searchableText(v))
\]

于是得到：

\[
S_0=\{v\mid match(v,text_s)\}
\]

\[
T_0=\{v\mid match(v,text_t)\}
\]

即：

```text
S0：起点实体集合
T0：重点/终点实体集合
```

这是 OR 语义：包含任意一个关键词即可，不要求包含全部关键词。

每个匹配实体必须记录：

```text
entity_id
matched_keyword
matched_field
matched_text
match_method
```

如果 `S0` 或 `T0` 为空，构建必须失败，不能自动选择其他实体替代。

同一实体可以同时出现在 `S0` 和 `T0` 中。后续需要将它表示成两个不同的 PNet 节点实例，避免层级冲突。

---

## 3. 双端 BFS

## 3.1 起点侧 BFS

从全部起点实体同时执行确定性多源 BFS：

\[
d_s(v)=\min_{s\in S_0} dist(s,v)
\]

起点侧第 \(i\) 层为：

\[
S_i=\{v\mid d_s(v)=i\},\quad 0\le i\le H_s
\]

起点侧层级顺序为：

```text
S0 → S1 → S2 → ... → SHs
```

其中：

```text
S0：关键词直接匹配出的起点实体
SHs：起点侧最外层前沿
```

每个实体只属于其最短距离对应的 BFS 层。若同一实体可以从多个起点以相同最短距离到达，只保留一个实体节点，但保存所有等长父边。

## 3.2 终点侧 BFS

从全部重点实体同时执行确定性多源 BFS：

\[
d_t(v)=\min_{t\in T_0} dist(v,t)
\]

搜索时仍然从 `T0` 向外扩展，得到：

\[
T_i=\{v\mid d_t(v)=i\},\quad 0\le i\le H_t
\]

但是在 PNet 中，终点侧必须倒序排列：

```text
THt → T(Ht-1) → ... → T1 → T0
```

因此 `T0` 位于整个 PNet 的最后一层。

---

## 4. BFS 内部边的选择

不建议只保留 BFS 首次发现时的一棵树，因为这样会丢失同层实体之间的多父节点知识结构。

建议保留所有连接相邻 BFS 深度的 KG 邻接关系。

### 4.1 起点侧

若两个实体通过 KG 关系相邻，并满足：

\[
d_s(v)=d_s(u)+1
\]

则生成 PNet 边：

```text
u → v
```

即：

```text
Si → S(i+1)
```

### 4.2 终点侧

若两个实体通过 KG 关系相邻，并满足：

\[
d_t(v)=d_t(u)+1
\]

由于终点侧需要倒序传播，生成 PNet 边：

```text
v → u
```

即：

```text
T(i+1) → Ti
```

### 4.3 不进入 PNet 的 KG 边

以下关系不进入最终 PNet：

- 同一个 BFS 层内的边；
- 跨越两个或更多 BFS 层的边；
- 与 PNet 计算方向相反的回边；
- 自环；
- 重复的 source-target 边。

这些关系可以写入审计文件，但不能写入最终 `edges.tsv`。

如果同一对 PNet 节点之间存在多条 KG 关系，必须合并为一条 PNet 边，因为当前图加载器不允许重复的 source-target 节点对。所有原始关系 ID 使用 `|` 连接，写入：

```text
evidence_relation_ids
```

---

## 5. 最外层全连接

令：

```text
FS = SHs
FT = THt
```

对两个最外层前沿执行笛卡尔积全连接：

\[
E_{bridge}=F_S\times F_T
\]

即对任意：

```text
u ∈ FS
v ∈ FT
```

生成：

```text
u → v
```

因此桥接边数量必须严格等于：

\[
|E_{bridge}|=|F_S|\times|F_T|
\]

桥接边不是知识图谱中的医学关系，必须显式标记为人工结构边：

```text
relation_type = structural_frontier_bridge
knowledge_source = dual_bfs_algorithm
evidence_relation_ids = 空
```

建议：

```text
confidence = 1.0
is_fallback = false
```

这里的 `confidence=1.0` 表示“确定执行了算法规定的结构连接”，不是医学关系置信度。

可以额外增加加载器忽略但审计程序使用的字段：

```text
edge_kind = structural_bridge
is_structural = true
```

不得将桥接边描述为实体之间存在真实的医学关系或因果关系。

由于全连接边数量可能很大，外部构建器应设置：

```text
max_bridge_edges
```

如果：

\[
|F_S|\times|F_T|>max\_bridge\_edges
\]

正式构建应直接失败，不能静默抽样，否则结果不再是“末端层级全连接”。

---

## 6. 完整 PNet 层级

最终层顺序为：

```text
S0 → S1 → ... → SHs → THt → ... → T1 → T0
```

全局层编号可以定义为：

\[
layer(S_i)=i
\]

\[
layer(T_j)=H_s+1+(H_t-j)
\]

因此总层数为：

\[
H_s+H_t+2
\]

例如：

```text
source_max_hops = 2
target_max_hops = 2
```

最终层级是：

```text
order 0：source_d000
order 1：source_d001
order 2：source_d002
order 3：target_d002
order 4：target_d001
order 5：target_d000
```

计算路径为：

```text
起点关键词实体
  → 起点侧 BFS
  → 起点侧最外层
  → 全连接结构桥
  → 终点侧最外层
  → 反向 BFS 层级
  → 重点关键词实体
```

---

## 7. 实体重叠和节点 ID

KG 实体 ID 与 PNet 节点 ID 必须区分。

一个 KG 实体可能：

- 同时匹配 `text_s` 和 `text_t`；
- 在起点侧和终点侧 BFS 中同时出现；
- 因结构补齐而在多个 PNet 层出现。

因此不能直接把 `entity_id` 当作全局唯一的 `node_id`。

建议节点 ID 格式：

```text
s::d000::<encoded_entity_id>
s::d001::<encoded_entity_id>

t::d002::<encoded_entity_id>
t::d001::<encoded_entity_id>
t::d000::<encoded_entity_id>
```

例如，同一个实体 `EFO_0000400` 出现在两侧时：

```text
s::d001::EFO_0000400
t::d000::EFO_0000400
```

两者具有不同的 `node_id`，但相同的：

```text
entity_id = EFO_0000400
```

这样可以防止同一实体被错误地分配到两个 PNet 层。

---

## 8. BFS 提前终止与 carry 节点

某些实体分支可能在到达配置的最大跳数前就没有邻居。如果直接保留这些节点，它们可能无法到达 PNet 输出层，不能通过当前项目的图校验。

建议使用透明的结构补齐节点，而不是删除关键词匹配实体。

假设某个起点侧分支在深度 1 终止，但 `source_max_hops=3`，则生成：

```text
真实节点@S1
  → carry@S2
  → carry@S3
```

终点侧同理，但 PNet 边方向倒置：

```text
carry@T3
  → carry@T2
  → 真实节点@T1
  → T0
```

carry 节点必须标记为：

```text
node_type = structural_carry
is_fallback = true
source = dual_bfs_algorithm
```

carry 边必须标记为：

```text
relation_type = structural_carry
knowledge_source = dual_bfs_algorithm
confidence = 1.0
is_fallback = true
evidence_relation_ids = 空
```

carry 节点只是层级对齐结构，解释模型结果时应折叠或排除，不能作为医学知识节点。

如果外部项目明确采用“删除未到达最外层的分支”策略，也可以不生成 carry，但必须在 manifest 中声明：

```text
dead_end_policy = prune
```

为保证所有关键词匹配实体得到保留，推荐默认使用：

```text
dead_end_policy = structural_carry
```

---

# 9. 必须产出的文件

能够被当前项目直接加载的最小文件集合是：

```text
pnet_output/
├── graph.yaml
├── nodes.tsv
└── edges.tsv
```

建议完整交付集合为：

```text
pnet_output/
├── graph.yaml
├── nodes.tsv
├── edges.tsv
├── entity_matches.tsv
├── bfs_occurrences.tsv
├── rejected_edges.tsv
└── build_manifest.json
```

前三个是模型输入；后四个用于复现和审计。

---

## 10. `nodes.tsv`

### 10.1 当前项目识别的字段

```tsv
node_id	entity_id	layer	node_type	display_name	is_fallback	source	external_id	original_node_id	description
```

字段说明：

| 字段 | 必需 | 说明 |
|---|---:|---|
| `node_id` | 是 | PNet 中全局唯一的节点实例 ID |
| `entity_id` | 是 | 原始 KG 实体 ID；结构节点使用合成 ID |
| `layer` | 是 | 必须与 `graph.yaml` 中的层 ID 一致 |
| `node_type` | 是 | 原始实体类型或 `structural_carry` |
| `display_name` | 是 | 显示名称，不能为空 |
| `is_fallback` | 是 | 真实实体为 `false`，carry 节点为 `true` |
| `source` | 是 | 如 `OptimusKG` 或 `dual_bfs_algorithm` |
| `external_id` | 否 | 原始外部 KG ID |
| `original_node_id` | 否 | carry 节点引用其对应真实节点或前一节点 |
| `description` | 否 | 实体描述 |

可以附加以下审计字段，当前加载器会忽略：

```text
side
bfs_depth
matched_keywords
is_structural
```

### 10.2 示例

```tsv
node_id	entity_id	layer	node_type	display_name	is_fallback	source	external_id	original_node_id	description	side	bfs_depth
s::d000::E1	E1	source_d000	clinical_concept	Heart sound	false	OptimusKG	E1			source	0
s::d001::E2	E2	source_d001	phenotype	Abnormal heart sound	false	OptimusKG	E2			source	1
s::d002::E3	E3	source_d002	disease	Cardiovascular disorder	false	OptimusKG	E3			source	2
t::d002::E4	E4	target_d002	gene	Example gene	false	OptimusKG	E4			target	2
t::d001::E5	E5	target_d001	phenotype	Hyperglycemia	false	OptimusKG	E5			target	1
t::d000::EFO_0000400	EFO_0000400	target_d000	disease	Diabetes mellitus	false	OptimusKG	EFO_0000400			target	0
```

---

## 11. `edges.tsv`

### 11.1 当前项目识别的字段

```tsv
source_node_id	target_node_id	relation_type	knowledge_source	confidence	is_fallback	evidence_relation_ids	enabled
```

字段说明：

| 字段 | 必需 | 说明 |
|---|---:|---|
| `source_node_id` | 是 | 较前一 PNet 层的节点 |
| `target_node_id` | 是 | 紧邻后一 PNet 层的节点 |
| `relation_type` | 是 | KG relation、`structural_frontier_bridge` 或 `structural_carry` |
| `knowledge_source` | 是 | KG 名称或 `dual_bfs_algorithm` |
| `confidence` | 是 | `[0,1]` 浮点数 |
| `is_fallback` | 是 | carry 边为 `true`，正常 KG/bridge 边为 `false` |
| `evidence_relation_ids` | 否 | 多个原始关系 ID 使用 `|` 分隔 |
| `enabled` | 否 | 默认为 `true`；`false` 的边不进入模型 |

可附加以下审计字段：

```text
edge_kind
is_structural
original_source_entity_id
original_target_entity_id
original_direction
traversal_direction
```

### 11.2 示例

```tsv
source_node_id	target_node_id	relation_type	knowledge_source	confidence	is_fallback	evidence_relation_ids	enabled	edge_kind
s::d000::E1	s::d001::E2	has_phenotype	OptimusKG	0.95	false	R1001	true	kg_bfs
s::d001::E2	s::d002::E3	associated_with	OptimusKG	0.90	false	R1002|R1003	true	kg_bfs
s::d002::E3	t::d002::E4	structural_frontier_bridge	dual_bfs_algorithm	1.0	false		true	structural_bridge
t::d002::E4	t::d001::E5	kg_bfs_reverse	OptimusKG	0.88	false	R2001	true	kg_bfs
t::d001::E5	t::d000::EFO_0000400	associated_with	OptimusKG	0.93	false	R2002	true	kg_bfs
```

在 `undirected` 搜索模式下，`source_node_id → target_node_id` 是 PNet 计算方向，不一定等于原始 KG 关系方向。因此原始方向必须保存在审计字段或 `bfs_occurrences.tsv` 中。

---

## 12. `graph.yaml`

示例使用 `Hs=2`、`Ht=2`：

```yaml
schema_version: pnet-graph-v1

layers:
  - id: source_d000
    order: 0
  - id: source_d001
    order: 1
  - id: source_d002
    order: 2
  - id: target_d002
    order: 3
  - id: target_d001
    order: 4
  - id: target_d000
    order: 5

nodes_file: nodes.tsv
edges_file: edges.tsv
```

外部项目可以增加元数据，当前加载器会忽略不认识的顶层字段。例如：

```yaml
algorithm:
  name: dual_keyword_bfs_frontier_bridge
  version: "1.0"
  traversal_mode: undirected
  source_max_hops: 2
  target_max_hops: 2
  keyword_match: normalized_substring_any
  dead_end_policy: structural_carry
  bridge_policy: complete_bipartite
```

模型真正使用的是：

```text
schema_version
layers
nodes_file
edges_file
```

---

## 13. `entity_matches.tsv`

该文件用于证明关键词是如何变成起点和终点实体的。

建议字段：

```tsv
side	keyword	normalized_keyword	entity_id	display_name	matched_field	matched_text	match_method
```

示例：

```tsv
side	keyword	normalized_keyword	entity_id	display_name	matched_field	matched_text	match_method
source	heart sound	heart sound	E1	Heart sound	display_name	Heart sound	normalized_substring
target	diabetes	diabetes	EFO_0000400	Diabetes mellitus	display_name	Diabetes mellitus	normalized_substring
```

如果一个实体匹配多个关键词，应保留多行，不要只保留第一个匹配结果。

---

## 14. `bfs_occurrences.tsv`

用于记录 BFS 过程及节点实例映射。

建议字段：

```tsv
side	node_id	entity_id	bfs_depth	is_seed	is_carry	parent_node_ids	parent_relation_ids
```

其中多个父节点或关系 ID 使用 `|` 分隔。

示例：

```tsv
side	node_id	entity_id	bfs_depth	is_seed	is_carry	parent_node_ids	parent_relation_ids
source	s::d000::E1	E1	0	true	false		
source	s::d001::E2	E2	1	false	false	s::d000::E1	R1001
source	s::d002::E3	E3	2	false	false	s::d001::E2	R1002|R1003
target	t::d000::EFO_0000400	EFO_0000400	0	true	false		
target	t::d001::E5	E5	1	false	false	t::d000::EFO_0000400	R2002
```

这里的 `parent_node_ids` 表示 BFS 搜索树方向；终点侧最终 PNet 边方向与它相反。

---

## 15. `rejected_edges.tsv`

建议记录所有被发现但没有进入最终 PNet 的关系：

```tsv
side	source_entity_id	target_entity_id	relation_id	source_depth	target_depth	rejection_reason
```

`rejection_reason` 建议取值：

```text
same_layer
backward
depth_gap
outside_hop_limit
self_loop
duplicate_pair
filtered_relation_type
filtered_entity_type
```

---

## 16. `build_manifest.json`

建议结构：

```json
{
  "schema_version": "dual-bfs-pnet-manifest-v1",
  "algorithm": {
    "name": "dual_keyword_bfs_frontier_bridge",
    "version": "1.0",
    "keyword_match_rule": "normalized_substring_any",
    "traversal_mode": "undirected",
    "source_max_hops": 2,
    "target_max_hops": 2,
    "dead_end_policy": "structural_carry",
    "bridge_policy": "complete_bipartite",
    "deterministic_sort": "entity_id_then_relation_id"
  },
  "inputs": {
    "kg_name": "OptimusKG",
    "kg_version": "填写实际版本",
    "kg_content_hash": "填写输入知识图谱哈希",
    "text_s": [
      "heart sound"
    ],
    "text_t": [
      "diabetes"
    ]
  },
  "matching": {
    "source_matched_entity_count": 5,
    "target_matched_entity_count": 3
  },
  "graph": {
    "source_frontier_count": 20,
    "target_frontier_count": 12,
    "expected_bridge_edge_count": 240,
    "actual_bridge_edge_count": 240,
    "node_count": 86,
    "edge_count": 412,
    "layer_widths": [
      5,
      17,
      20,
      12,
      21,
      11
    ],
    "carry_node_count": 4,
    "carry_edge_count": 4
  },
  "limits": {
    "max_nodes": 100000,
    "max_edges": 1000000,
    "max_bridge_edges": 500000
  },
  "status": {
    "search_complete": true,
    "bridge_complete": true,
    "validation_passed": true
  }
}
```

不得将因达到节点、边或时间预算而被截断的结果标记为：

```json
"search_complete": true
```

正式训练只应接收：

```text
search_complete = true
bridge_complete = true
validation_passed = true
```

---

## 17. 确定性要求

为了让外部项目重复运行得到相同结果，应规定：

1. 关键词规范化算法固定；
2. 匹配结果按 `entity_id` 排序；
3. BFS 队列按 `entity_id` 排序；
4. 邻边按 `neighbor_entity_id, relation_id` 排序；
5. 同一实体始终取最短 BFS 深度；
6. 等长父节点全部保留；
7. 同一 PNet source-target 对只生成一条边；
8. 多关系 ID 排序后用 `|` 拼接；
9. `nodes.tsv` 按 `layer order, node_id` 排序；
10. `edges.tsv` 按 `source_node_id, target_node_id` 排序。

---

## 18. 最终验收条件

外部项目产出的 PNet 至少应满足：

- `text_s`、`text_t` 均非空；
- 起点和终点关键词均至少匹配一个实体；
- 每个 PNet 层非空；
- `node_id` 全局唯一；
- 所有边的端点均存在；
- 所有边只连接相邻层；
- 无自环；
- 无重复 source-target 边；
- 整张图为 DAG；
- 每个节点都能到达最终 `target_d000` 层；
- 桥接边数量严格等于两个前沿节点数的乘积；
- 真实 KG 边具有可追溯的 `evidence_relation_ids`；
- bridge/carry 边明确标记为人工结构边；
- 同一输入重复构建得到相同的节点、边和层顺序。

最终模型所消费的结构可以概括为：

```text
text_s
  → 起点实体 S0
  → 正向 BFS 分层
  → 起点前沿 SHs
  → 完全二部结构桥
  → 终点前沿 THt
  → 倒序 BFS 分层
  → 重点实体 T0
```

其中 KG 边提供可解释的局部知识结构；前沿全连接边只负责把两个独立搜索空间组合成完整 PNet，不应被解释成新增的医学知识关系。