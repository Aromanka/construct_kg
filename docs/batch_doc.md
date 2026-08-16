# OpenAlex 分批筛选与全文知识抽取

本流程严格分为两个阶段：

1. 流式扫描 OpenAlex Works 元数据，筛查并持久化候选 Work ID；
2. 仅为已选 Work 获取全文，再调用 LLM 进行知识抽取。

普通 `data/works/*.gz` 只提供 Work 元数据、摘要、正文可用性提示和 URL，不包含完整
正文。项目已经内置 OpenAlex 官方内容服务下载能力：直接访问
`https://content.openalex.org`，按 Work ID 尝试 GROBID XML 和 PDF，并支持
`OPENALEX_API_KEY`。当前实现不调用 OpenAlex CLI，也不使用 `api.openalex.org` 元数据
API 下载正文；同时不会自动访问 publisher 的第三方 PDF URL。

## 0. 环境变量

以下示例以 `batch_000` 为当前元数据批次，所有批次共用同一个 workspace：

```bash
BATCH=/home/bml/storage/mnt/v-vmkfid4oobb3c0qh/xdiabetes/openalex
# WORKSPACE=data/openalex
REMOTE=PatrickGuan@47.116.73.1:/volume1/X-Diabetes/openalex/openalex-snapshot/data/works
```

将 `user@ip` 改为实际值。

## 1. 筛查 Work ID

### 1.1 下载当前 Works 元数据批次

```bash
mkdir -p "$BATCH/data/works"

scp -P 22222 -r \
  "${REMOTE}/updated_date=2016-*" \
  "$BATCH/data/works/"
scp -P 22222 -r   "${REMOTE}/updated_date=2025-10*"   "$BATCH/data/works/"
```

current progress to 2025-10

该命令只复制 Works 元数据，不复制全文。不要复制 `legacy-data` 或异常后缀文件。

### 1.2 检查批次

```bash
find "$BATCH/data/works" -mindepth 1 -maxdepth 1 -type d | sort
find "$BATCH/data/works" -type f -name 'part_[0-9][0-9][0-9][0-9].gz' | wc -l
du -sh "$BATCH"
```

### 1.3 执行确定性筛选

```bash
python openalex_pipeline.py select "$BATCH" \
  --keyword "<关键词1>" \
  --keyword-mode any \
  --require-fulltext \
  --workspace "$WORKSPACE"
```

该命令按以下顺序处理：

1. 默认 OpenAlex Field 顶层门控，仅保留项目配置的医学和生物医学领域；
2. 在标题与摘要中匹配关键词；
3. `--require-fulltext` 根据元数据提示，优先保留可能具有全文的 Work；
4. 将选中状态、Work ID、原始 Work JSON 和快照位置写入
   `$WORKSPACE/catalog.sqlite3`。

`--require-fulltext` 只是元数据预筛，不保证 OpenAlex 内容服务一定能返回正文。需要
source 限制时追加 `--source "<source 名称或 ID>"`。若有多个元数据批次，逐批执行上述
`select` 命令，并始终使用同一个 `--workspace`；Work ID 会幂等更新。

如需导出已选 Work ID，可执行：

```bash
sqlite3 "$WORKSPACE/catalog.sqlite3" \
  "SELECT work_id FROM works WHERE selected=1 ORDER BY work_id;" \
  > "$WORKSPACE/selected_work_ids.txt"
```

## 2. 获取全文并进行 LLM 知识抽取

候选批次全部筛查完成后再执行本阶段。执行 `materialize` 前，应保留一个结构有效的
`$BATCH/data/works` 目录；已选 Work 的完整元数据已经保存在公共 catalog 中。

### 2.1 从 OpenAlex 官方内容服务获取全文

```bash
export OPENALEX_API_KEY="<你的 OpenAlex API Key>"

python openalex_pipeline.py materialize "$BATCH" \
  --workspace "$WORKSPACE" \
  --download-fulltext \
  --content-mode fulltext
```

程序会对 catalog 中已选 Work 依次尝试：

```text
https://content.openalex.org/works/W....grobid-xml
https://content.openalex.org/works/W....pdf
```

下载文件保存在 `$WORKSPACE/fulltext/`，解析后的知识抽取输入保存在
`$WORKSPACE/documents/W....json`。`--content-mode fulltext` 不允许使用摘要兜底；无法取得
或解析全文的 Work 会标记为 `not_found` 并跳过，不会进入 LLM 抽取。

如果全文已经由其他方式下载，可改用本地目录，不启用网络下载：

```bash
python openalex_pipeline.py materialize "$BATCH" \
  --workspace "$WORKSPACE" \
  --fulltext-dir /path/to/openalex-fulltext \
  --content-mode fulltext
```

本地文件名必须使用稳定 Work ID，例如 `W123.xml`、`W123.grobid-xml`、`W123.txt` 或
`W123.pdf`。解析 PDF 需要安装 `medical-kg[pdf]` 可选依赖。

### 2.2 对成功取得的全文执行 LLM 知识抽取

```bash
python -m medical_kg run "$WORKSPACE/documents" \
  --source-type research \
  --chunk-size 12000 \
  --chunk-overlap 500 \
  --config config.yaml
```

该命令会将生成的 `W....json` 文档注册到主知识库，然后调用 `config.yaml` 配置的 LLM，
执行 general、molecular 和 clinical 抽取流程。结果写入 `config.yaml` 指定的主 SQLite
知识库；OpenAlex 候选与全文状态仍保存在 `$WORKSPACE/catalog.sqlite3`。

流水线具有幂等和断点续跑能力：成功完成且处理版本未变化的文档不会重复抽取，失败任务
可通过以下命令重新入队：

```bash
python -m medical_kg retry-failed --config config.yaml
python -m medical_kg extract --config config.yaml
```

## 3. 清理时机

只有在以下条件全部满足后，才清理 `batch_000` 临时元数据目录：

- 所有元数据批次的 Work ID 筛查已经完成；
- 全文 materialize 阶段已经结束；
- `$WORKSPACE/catalog.sqlite3`、`$WORKSPACE/fulltext/` 和
  `$WORKSPACE/documents/` 已确认保存；
- LLM 抽取结果已写入主知识库或已完成可靠备份。
