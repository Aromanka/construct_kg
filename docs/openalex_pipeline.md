# OpenAlex 文献筛选与知识图谱抽取

核心入口是项目根目录的 `openalex_pipeline.py`。该入口适配以下快照布局：

```text
openalex-snapshot/
└── data/
    ├── works/
    │   ├── manifest
    │   └── updated_date=YYYY-MM-DD/part_XXXX.gz
    └── sources/
        ├── manifest
        └── updated_date=YYYY-MM-DD/part_XXXX.gz
```

程序会扫描所有 `updated_date=*` 分区，并且只接受严格匹配 `part_数字.gz` 的文件；
`part_0001.gz.ABC123` 一类异常副本不会被读取。gzip 文件按 JSONL 逐行解压，不会全量
解压快照。

## 一次运行

仅使用确定性过滤并准备文档：

```powershell
E:\code\Env\envs\ml_env\python.exe openalex_pipeline.py run D:\openalex-snapshot `
  --keyword diabetes `
  --keyword "voice analysis" `
  --keyword-mode any `
  --exclude-keyword review `
  --source "Journal of Voice" `
  --require-fulltext `
  --fulltext-dir D:\openalex-fulltext `
  --content-mode fulltext-or-abstract `
  --enrich-sources `
  --workspace data/openalex
```

加入 LLM 筛选：

```powershell
E:\code\Env\envs\ml_env\python.exe openalex_pipeline.py run D:\openalex-snapshot `
  --keyword diabetes `
  --llm-prompt "@prompts/my_openalex_screening.txt" `
  --llm-batch-size 20 `
  --content-mode abstract `
  --workspace data/openalex `
  --config config.yaml
```

每个 LLM 请求包含一批带有 `[1]`、`[2]` 等编号的标题和摘要。模型必须返回：

```json
{"selected_numbers": [1, 4, 7]}
```

筛选提示可直接写在 `--llm-prompt` 后，也可用 `@文件路径` 读取。LLM 配置沿用项目
`config.yaml` 中的 OpenAI-compatible/DeepSeek 配置。

若要在文档准备完成后立即抽取知识关系，增加：

```powershell
--extract --config config.yaml
```

这会复用项目现有的分块、并发 LLM 抽取、断点续跑、Bronze/Silver/Gold 数据结构，
不会另建一套关系格式。

## 筛选规则

- 默认顶层 Field 门控：从 Work 的 `topics[].field.id` 读取 OpenAlex Field，只保留
  Medicine (27)、Health Professions (36)、Biochemistry, Genetics and Molecular Biology (13)、
  Immunology and Microbiology (24)、Neuroscience (28)、Pharmacology, Toxicology and
  Pharmaceutics (30)。缺少 Field 的 Work 默认丢弃。
- `--field ID`：可重复；用指定 OpenAlex Field ID 替换上述默认集合。
- `--no-medical-field-filter`：关闭默认 Field 门控；若没有其他筛选条件，还需显式指定
  `--all`。
- `--keyword`：可重复；匹配标题与恢复后的摘要，不区分大小写并做 Unicode NFKC 归一化。
- `--keyword-mode any|all`：任一关键词命中或所有关键词同时命中。
- `--exclude-keyword`：命中任一排除词即丢弃。
- `--source`：可重复；匹配 Work 内所有 `locations[].source` 的 ID、名称、类型等短字段。
- `--require-fulltext`：要求 Work 元数据具有 `has_content`、`has_fulltext` 或 `pdf_url` 提示。
- `--llm-prompt`：在上述低成本过滤之后，批量筛查候选标题和摘要。
- `--include-id W...`：无视普通过滤条件，始终加入指定 Work。
- `--max-candidates` / `--max-selected`：控制候选或选中数量。
- `--max-works`：仅用于小规模试跑；正式全快照筛选不要设置。
- `--all`：选取 Field 门控放行的全部 Work；若同时使用 `--no-medical-field-filter`，则明确
  允许无过滤条件选取全部 Work。

`--require-fulltext` 只说明快照元数据提示“可能有正文”，不等于正文已经下载或解析。
若必须拥有实际正文，使用 `--content-mode fulltext`；无法解析到正文的 Work 会记录在 catalog，
但不会生成待抽取 JSON 文档。

## 正文与摘要

摘要由 `abstract_inverted_index` 按位置恢复，并始终单独保存为 `abstract`。OpenAlex metadata
snapshot 本身不是全文库，正文通过以下方式解析：

1. `--fulltext-dir` 指定的本地目录；
2. 显式使用 `--download-fulltext` 后，从 `content.openalex.org` 获取选中 Work 的 GROBID XML
   或 PDF；需要凭证时通过 `--openalex-api-key` 或 `OPENALEX_API_KEY` 提供。

本地正文采用稳定 Work ID 文件名，支持：

```text
W123.grobid-xml
W123.xml
W123.xml.gz
W123.txt
W123.md
W123.json
W123.pdf
W123/W123.xml
```

PDF 解析需要安装项目的 `pdf` 可选依赖。程序不会自动抓取任意 publisher `pdf_url`，避免
未经控制地访问快照内第三方 URL；远程下载仅访问 OpenAlex content 主机。

`--content-mode` 有三种模式：

- `fulltext`：只生成实际取得正文的文档；
- `abstract`：仅使用摘要，适合先做小成本关系抽取；
- `fulltext-or-abstract`：优先正文，无正文时用摘要。

## 稳定编号、增量追加与检索

筛选结果保存在 `<workspace>/catalog.sqlite3`。主键是 OpenAlex `W...` Work ID，知识库
document ID 是 `openalex:W...`。每条记录还保留：

- 标题、恢复后的摘要、DOI、年份、语言与类型；
- primary source 与所有 location sources；
- 全文可用性提示、URL、解析状态与本地路径；
- 命中关键词、筛选方式；
- 完整原始 Work JSON；
- 原始 gzip 相对路径和 JSONL 行号。

因此未来增加字段时，可直接从 `raw_json` 获取，或依靠 `snapshot_file + snapshot_line`
回溯原始快照。

显式增加文档：

```powershell
E:\code\Env\envs\ml_env\python.exe openalex_pipeline.py add D:\openalex-snapshot `
  W2741809807 W1234567890 --workspace data/openalex
```

`add` 加入的 Work 标记为 `manual`，之后重新执行普通筛选不会取消它。读取某一条完整记录：

```powershell
E:\code\Env\envs\ml_env\python.exe openalex_pipeline.py show W2741809807 `
  --workspace data/openalex
```

只重新准备 catalog 中已经选中的文档，无需再次筛选：

```powershell
E:\code\Env\envs\ml_env\python.exe openalex_pipeline.py materialize D:\openalex-snapshot `
  --workspace data/openalex `
  --fulltext-dir D:\openalex-fulltext `
  --content-mode fulltext
```

## 输出

```text
data/openalex/
├── catalog.sqlite3
├── documents/
│   └── W....json
└── fulltext/
    ├── W....grobid-xml
    └── W....pdf
```

文档 JSON 同时包含 `abstract`、`full_text`、最终交给抽取器的 `content`、sources、引用
Work IDs 和快照定位信息。关系抽取结果仍写入项目 `config.yaml` 指定的主 SQLite 数据库。
