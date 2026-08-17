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
解压快照。合法命名但被截断或损坏的 gzip 分片会记录错误并跳过，后续分片继续处理；
最终 `selection.snapshot_failures` 会列出路径、异常类型和损坏前读取的记录数。若任务要求
任一分片损坏都立即终止，增加 `--strict-snapshot`。

跳过损坏分片只能保证批处理继续运行，不能恢复其中缺失的论文。应使用 `gzip -t` 定位并从
原始快照重新复制损坏文件，然后利用 catalog 的幂等更新能力重新运行筛选。

## 一次运行

仅使用确定性过滤并准备文档：

```bash
python openalex_pipeline.py run /home/bml/storage/mnt/v-vmkfid4oobb3c0qh/xdiabetes/openalex \
  --keyword diabetes \
  --keyword "voice" \
  --keyword-mode any \
  --exclude-keyword review \
  --require-abstract \
  --enrich-sources \
  --workspace data/openalex
```

对于增量：
```bash
python openalex_pipeline.py select /home/bml/storage/mnt/v-vmkfid4oobb3c0qh/xdiabetes/openalex \
  --keyword diabetes \
  --keyword "voice" \
  --keyword-mode any \
  --exclude-keyword review \
  --require-abstract \
  --enrich-sources \
  --workspace data/openalex
```

`run` 和 `materialize` 默认采用 `fulltext` 模式，并默认从 OpenAlex 内容服务下载正文；
仅使用本地正文时增加 `--no-download-fulltext`。

### 已完成筛选后，仅从摘要抽取知识

如果已经执行完上面的 `run`，筛选结果已保存在 `data/openalex/catalog.sqlite3`，无需再次扫描
OpenAlex 快照。先把 catalog 中已选文章的摘要准备成抽取文档：

```bash
python openalex_pipeline.py materialize /home/bml/storage/mnt/v-vmkfid4oobb3c0qh/xdiabetes/openalex \
  --workspace data/openalex \
  --content-mode abstract \
  --abstract-chunk-size 12000 \
  --no-download-fulltext
```

然后只对摘要批次目录执行知识抽取：

```bash
python -m medical_kg run data/openalex/documents/abstract_batches \
  --source-type research \
  --config config.yaml
```

第二条命令的输入目录仅包含由 OpenAlex 摘要生成的文档，因此不会读取此前准备的全文文档。
关系抽取结果写入 `config.yaml` 配置的主 SQLite 数据库。上述命令可断点续跑；已经成功入库
和抽取且内容未变化的文档会被跳过。

加入 LLM 筛选：

```bash
python openalex_pipeline.py run /home/bml/storage/mnt/v-vmkfid4oobb3c0qh/xdiabetes/openalex \
  --keyword diabetes \
  --llm-prompt "@prompts/my_openalex_screening.txt" \
  --llm-batch-size 20 \
  --content-mode abstract \
  --abstract-chunk-size 12000 \
  --workspace data/openalex \
  --config config.yaml
```

每个 LLM 请求包含一批带有 `[1]`、`[2]` 等编号的标题和摘要。模型必须返回：

```json
{"selected_numbers": [1, 4, 7]}
```

筛选提示可直接写在 `--llm-prompt` 后，也可用 `@文件路径` 读取。LLM 配置沿用项目
`config.yaml` 中的 OpenAI-compatible/DeepSeek 配置。

如果尚未执行筛选，并希望在同一次 `run` 中仅准备摘要且立即抽取知识关系，在筛选参数后增加：

```bash
  --content-mode abstract \
  --no-download-fulltext \
  --extract \
  --config config.yaml
```

这会复用项目现有的分块、并发 LLM 抽取、断点续跑、Bronze/Silver/Gold 数据结构，不会另建
一套关系格式。若筛选已经完成，优先使用上面的“两步命令”，避免重新扫描快照。

## 生成 Gold 图谱并可视化

OpenAlex 抽取结果首先写入主 SQLite 数据库的 Bronze 层。完成抽取后，运行以下命令进行实体消歧、
关系规范化并生成 Gold 图谱：

```bash
python -m medical_kg canonicalize --config config.yaml
```

`canonicalize` 读取 `config.yaml` 中的 `database.path`。该路径必须与前面 OpenAlex 抽取实际写入的
数据库一致；使用项目默认数据库 `data/medical_kg.sqlite3` 时，应确保配置为：

```yaml
database:
  path: data/medical_kg.sqlite3
```

启动本地 Neo4j 后，只将同一个 SQLite 数据库中的 Gold 表导入 Neo4j，并启动项目自带的 Web
可视化页面。该脚本不会导入 Bronze 原始关系、文档正文和任务记录，适合大型 OpenAlex 数据库：

```bash
python src/utils/sqlite_gold_to_neo4j.py all --sqlite data/medical_kg.sqlite3 --no-auth --clear --open-browser
```

页面默认打开 `http://127.0.0.1:8000`，其中默认视图为 Gold 规范化图。上述命令适用于已关闭
Neo4j 身份验证的本地实例；若启用了身份验证，应去掉 `--no-auth`，并通过
`NEO4J_PASSWORD` 环境变量或 `--password` 提供密码。转换脚本不会读取 `config.yaml`，所以
`--sqlite` 必须与 `database.path` 指向同一个文件。

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
- `--require-abstract`：要求 Work 的 `abstract_inverted_index` 能恢复出非空摘要。
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

每次 `select` 或 `run` 的 `selection` 输出都会报告选中文献的内容覆盖统计：
`selected_with_abstract`、`selected_without_abstract`、`selected_with_fulltext_hint`、
`selected_without_fulltext_hint` 和 `selected_with_abstract_and_fulltext_hint`。其中正文统计是
OpenAlex 元数据提示；实际正文取得数量以 materialization 输出的 `fulltext` 为准。

## 正文与摘要

摘要由 `abstract_inverted_index` 按位置恢复，并始终单独保存为 `abstract`。OpenAlex metadata
snapshot 本身不是全文库，正文通过以下方式解析：

1. `--fulltext-dir` 指定的本地目录；
2. 默认从 `content.openalex.org` 获取选中 Work 的 GROBID XML 或 PDF；需要凭证时通过
   `--openalex-api-key` 或 `OPENALEX_API_KEY` 提供。使用 `--no-download-fulltext` 可关闭下载。

额度不足、鉴权失败、限流、网络错误或单篇正文解析失败时，该 Work 会记录对应状态并跳过，
不会中断后续 Work。额度或凭证恢复后可重新执行，失败的 Work 不会被标为已处理。

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
- `abstract`：仅使用摘要；按 `--abstract-chunk-size`（默认 12000 字符）依次堆叠多篇
  文章，当前批次再加入一篇就会超限时才开始下一批，以减少短文本请求数；
- `fulltext-or-abstract`：优先正文，无正文时用摘要。

catalog 对两条处理路径分别维护 `fulltext_processed` / `abstract_processed` 和对应的
materialized path。成功准备摘要不会阻止之后处理同一 Work 的全文，反之亦然；重复执行同一
模式时会跳过已准备的内容。单篇摘要本身超过上限时会保持完整，交由后续字符分块继续拆分。

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

```bash
python openalex_pipeline.py add /home/bml/storage/mnt/v-vmkfid4oobb3c0qh/xdiabetes/openalex \
  W2741809807 W1234567890 --workspace data/openalex
```

`add` 加入的 Work 标记为 `manual`，之后重新执行普通筛选不会取消它。读取某一条完整记录：

```bash
python openalex_pipeline.py show W2741809807 \
  --workspace data/openalex
```

只重新准备 catalog 中已经选中的文档，无需再次筛选：

```bash
python openalex_pipeline.py materialize /home/bml/storage/mnt/v-vmkfid4oobb3c0qh/xdiabetes/openalex \
  --workspace data/openalex \
  --fulltext-dir /home/bml/storage/mnt/v-vmkfid4oobb3c0qh/xdiabetes/openalex-fulltext \
  --content-mode fulltext
```

## 输出

```text
data/openalex/
├── catalog.sqlite3
├── documents/
│   ├── W....json
│   └── abstract_batches/
│       └── abstract_batch_<hash>.json
└── fulltext/
    ├── W....grobid-xml
    └── W....pdf
```

文档 JSON 同时包含 `abstract`、`full_text`、最终交给抽取器的 `content`、sources、引用
Work IDs 和快照定位信息。关系抽取结果仍写入项目 `config.yaml` 指定的主 SQLite 数据库。
