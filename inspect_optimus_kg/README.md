# OptimusKG diabetes 子图（轻量 demo）

默认使用本机数据目录 `E:\code\data\knowledge\med\OptimusKG`，以
`EFO_0000400`（diabetes mellitus）为自动选择的唯一 seed，并只提取 seed
的一跳邻接节点。每类边默认最多 25 条，输出只包含分析常用的紧凑字段，
不会复制药物图片、长描述或其他大属性。

## 本机 demo

在项目根目录运行：

```powershell
& 'E:\code\Env\envs\ml_env\python.exe' .\inspect_optimus_kg\extract_diabetes_subgraph.py
```

默认输出到 `inspect_optimus_kg/output/diabetes_demo/`：

- `nodes.parquet`、`edges.parquet`：紧凑子图；
- `nodes.csv`、`edges.csv`：方便人工检查的小型预览；
- `disease_candidates.csv`：候选疾病及自动选择分数；
- `summary.json`：根节点、节点/边计数、自动校验结果及输出体积。

仅检查候选根节点：

```powershell
& 'E:\code\Env\envs\ml_env\python.exe' .\inspect_optimus_kg\01_find_diabetes.py
```

## 参数

```powershell
& 'E:\code\Env\envs\ml_env\python.exe' .\inspect_optimus_kg\extract_diabetes_subgraph.py `
  --kg-root 'E:\code\data\knowledge\med\OptimusKG' `
  --output-dir '.\inspect_optimus_kg\output\diabetes_demo' `
  --root-id EFO_0000400 `
  --max-edges-per-type 25
```

`--max-edges-per-type 0` 表示不限制每类边，适合服务器全量运行；本机不建议。
可用 `--edge-types` 只选部分关系，例如：

```powershell
--edge-types disease_disease disease_gene drug_disease
```

这里的“一跳”指所有输出边都直接连接 disease seed。不会从这些一跳邻居继续
扩展，因此不会意外形成二跳或更深的图遍历。
