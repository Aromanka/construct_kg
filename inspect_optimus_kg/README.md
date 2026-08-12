# OptimusKG diabetes 子图（轻量 demo）

默认使用本机数据目录 `E:\code\data\knowledge\med\OptimusKG`，自动选择
`EFO_0000400`（diabetes mellitus）作为唯一 seed。搜索深度可通过 `--hops`
在运行时设置，默认值为 1。

每一跳中的每类边默认最多保留 25 条。输出只包含分析常用的紧凑字段，不会
复制药物图片、长描述或其他大属性。

## 本机运行

默认提取 1-hop：

```powershell
& 'E:\code\Env\envs\ml_env\python.exe' `
  '.\inspect_optimus_kg\extract_diabetes_subgraph.py'
```

提取 2-hop，并限制每跳、每类关系最多 10 条边：

```powershell
& 'E:\code\Env\envs\ml_env\python.exe' `
  '.\inspect_optimus_kg\extract_diabetes_subgraph.py' `
  --hops 2 `
  --max-edges-per-type 10 `
  --output-dir '.\inspect_optimus_kg\output\diabetes_hop2_demo'
```

`--hops 0` 只输出 seed 节点，不提取边。`--hops 1` 保持原来的一跳行为。

默认输出目录是 `inspect_optimus_kg/output/diabetes_demo/`：

- `nodes.parquet`、`edges.parquet`：紧凑子图；
- `nodes.csv`、`edges.csv`：方便人工检查的小型预览；
- `disease_candidates.csv`：候选疾病及自动选择分数；
- `summary.json`：根节点、hop 数、节点/边计数、校验结果及输出体积。

`nodes` 中的 `hop_distance` 是节点到 seed 的最短发现距离；`edges` 中的
`discovered_hop` 表示该边在哪一轮 BFS 中首次被保留。

## 完整参数示例

```powershell
& 'E:\code\Env\envs\ml_env\python.exe' `
  '.\inspect_optimus_kg\extract_diabetes_subgraph.py' `
  --kg-root 'E:\code\data\knowledge\med\OptimusKG' `
  --output-dir '.\inspect_optimus_kg\output\diabetes_demo' `
  --root-id EFO_0000400 `
  --hops 1 `
  --max-edges-per-type 25
```

`--max-edges-per-type 0` 表示不限制边数。多跳搜索时子图可能快速膨胀，本机
不建议将它设为 0。也可以用 `--edge-types` 只选择部分关系：

```powershell
--edge-types disease_disease disease_gene drug_disease
```

Linux server run command
```bash
python inspect_optimus_kg/extract_diabetes_subgraph.py \
  --kg-root /data/home/wanglidi/code/p1/xdiabetes/data/OptimusKG/ \
  --output-dir outputs \
  --root-id EFO_0000400 \
  --max-edges-per-type 0 \
  --hops 2
```
