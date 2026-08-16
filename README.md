# Medical Literature Knowledge Graph

A maintainable, evidence-aware Python pipeline that converts local biomedical papers into a
resumable SQLite knowledge base. The implementation preserves a faithful Phase I Bronze layer,
then resolves reusable biomedical entities and relations into an auditable Silver/Gold graph with
multi-document evidence aggregation.

## What is implemented

- Stable document registration using relative paths (or a JSON-provided `document_id`) and
  SHA-256 content hashes, with immutable revision snapshots for historical provenance.
- Controlled document source types (`textbook`, `guidelines`, or `research`) propagated into
  extraction provenance and list-valued node/edge sources.
- Optional character-based document chunking with overlap, enabled only when requested.
- Text, Markdown, JSON and optional PDF input.
- Full-document `general`, `molecular`, and `clinical` extraction passes.
- Versioned YAML prompts and configurable relation vocabulary.
- Lightweight DeepSeek and third-party OpenAI-compatible HTTP client behind an `LLMClient`
  interface; the OpenAI SDK is not required.
- Pydantic validation, including controlled biomedical entity types, `OTHER` detail, extensible
  qualifiers, and confidence bounds.
- Semantics-preserving normalization for nullable empty qualifiers and one guided provider
  correction request for otherwise invalid structured output; strict validation still gates writes.
- SQLite persistence for all Landing, Bronze, Silver, and Gold objects described by the
  architecture specification.
- Atomic job claiming with process-local asynchronous write serialization and SQLite immediate
  transactions, independent failures, bounded
  concurrency, request/token rate limiting, bounded retries, and resume/retry commands.
- Exact source-evidence validation and deterministic mention offsets where possible.
- Structured logs and run-level request/token/success statistics.
- Executable Silver entity resolution with exact normalized aliases, type-scoped biomedical
  synonyms, abbreviation matching, lexical top-k candidate retrieval, and an optional contextual
  LLM decision for ambiguous candidates. Lexical similarity alone never merges entities.
- Deterministic relation canonicalization with optional LLM fallback to the controlled vocabulary.
- Gold assertion identity based on canonical subject, relation, object, semantic qualifiers,
  negation, and speculation; repeated facts aggregate into separate evidence records.
- Graph-quality metrics for singleton edges, largest connected component, canonical compression,
  cross-document reuse, `OTHER` relations, and duplicate fact support.

## Setup

Python 3.10+ in a Conda environment is required; SQLite is bundled with Python. The base installation
intentionally declares only the five direct runtime libraries needed by the core SQLite and
LLM pipeline.

```bash
conda create -n medical-kg python=3.10 -y
conda activate medical-kg
python -m pip install -e .
```

Copy `config.example.yaml` to `config.yaml` before running the project.

Set `DEEPSEEK_API_KEY`, or replace its environment placeholder in the local `config.yaml`. That
file is ignored by Git. Optional features are installed only when needed:

```bash
# PDF ingestion
python -m pip install -e ".[pdf]"

# Tests and linting
python -m pip install -e ".[dev]"
```

`requirements.txt` contains only core runtime dependencies. The CLI and retry behavior use the
Python standard library, avoiding separate CLI and retry-framework packages.

Create the schema and seed the configured relation vocabulary:

```bash
python -m medical_kg init-db --config config.yaml
```

The default database configuration uses a project-local file:

```yaml
database:
  path: data/medical_kg.sqlite3
  timeout: 30
  echo: false
```

Relative database paths are resolved from the directory containing `config.yaml`.

## Input

Plain text and Markdown files use their path relative to the ingested directory as the stable
document ID. A JSON file can contain either a string or this object:

```json
{
  "document_id": "PMID:123456",
  "title": "Example study",
  "doi": "10.0000/example",
  "pmid": "123456",
  "source_type": "research",
  "content": "Complete paper text..."
}
```

Changing the content under an existing `document_id` resets its processing jobs. Previous Bronze
extraction records and their exact source revision remain available for provenance.

## Commands

```bash
python -m medical_kg ingest ./papers
python -m medical_kg ingest ./guidelines --source-type guidelines
python -m medical_kg extract
python -m medical_kg run ./papers
python -m medical_kg run ./textbooks --source-type textbook --chunk-size 12000 --chunk-overlap 500
python -m medical_kg run --limit 100
python -m medical_kg extract --document-id PMID:123456
python -m medical_kg retry-failed
python -m medical_kg status
python -m medical_kg stats
python -m medical_kg canonicalize
python -m medical_kg canonicalize --semantic
```

Commands are executed through the Python module entry point; no generated `.exe` launcher is
required. Run them from the project root after activating the virtual environment.

### Guideline knowledge extraction

After the database has been initialized, use this command to ingest every document under
`data/knowledge_base` as `guidelines` and extract the knowledge graph with document chunking:

```powershell
python -m medical_kg run "data/knowledge_base" --source-type guidelines --chunk-size 12000 --chunk-overlap 500 --config config.yaml
```

PDF ingestion requires `python -m pip install -e ".[pdf]"` in the active Conda environment.
The SQLite database file and its parent directory are created automatically. The configured path
must be writable by the account running the pipeline.
The compatible client explicitly uses the operating system's trusted CA context, so an unrelated
or stale `SSL_CERT_FILE` override is not read during HTTP client initialization.

Every command accepts `--config PATH` where applicable. `run` optionally ingests a directory and
then extracts pending work. `--source-type` sets the default for ingested files; JSON metadata can
override it per document. `extract` and `run` use the complete document unless `--chunk-size` is
provided; `--chunk-overlap` must be smaller than the chunk size. A successful `(document_id,
extraction pass, effective stage version)` is skipped on subsequent runs. Chunk settings are part
of that effective version. Prompt versions are also included automatically, so prompt changes do
not reuse incompatible checkpoints. Increase `extraction.stage_version` when other extraction
semantics change.

### Parallel extraction

API requests use a bounded asynchronous queue and default to 100 concurrent connections. Chunked
documents execute their chunks concurrently, while SQLite-backed chunk checkpoints prevent
successful chunks from being requested again after an interruption. A small set of job claimers
feeds up to 100 active parent jobs, and job heartbeats prevent a long extraction from being
mistaken for an abandoned worker.

The relevant settings are under `processing` in `config.yaml`:

```yaml
job_claimers: 8
job_concurrency: 100
api_concurrency: 100
chunk_queue_size: 300
requests_per_minute: 100
tokens_per_minute: 1000000
distributed_rate_limit: true
```

Concurrency is only an upper bound. The shared SQLite rate limiter smooths requests across
processes using the same local database file, so actual in-flight requests also depend on provider
RPM/TPM limits and API latency. Writes are serialized before they acquire a pooled connection;
WAL mode and a busy timeout handle readers and competing processes. SQLite still has one writer at
a time, so keep the database on a local disk rather than NFS or another network filesystem. After
upgrading an existing checkout, run `python -m medical_kg init-db --config config.yaml` once to
create the current schema.

### Inspecting an interrupted build

The statistics command is read-only and reports committed documents, job states, completed
extraction passes, entity mentions, raw assertions, evidence validation, canonical graph counts,
and graph-quality metrics:

```powershell
python -m medical_kg stats --config config.yaml
```

`bronze_knowledge.raw_assertions` measures extracted surface assertions;
`canonical_knowledge.assertions` measures deduplicated facts and
`canonical_knowledge.evidence_links` measures their supporting raw assertions. A job left as
`RUNNING` after interruption is shown as-is; this inspection command never changes or resumes it.
Only completed extraction passes are committed, so an interrupted in-flight pass is not included
in the knowledge counts.

### Silver/Gold canonicalization

Run deterministic, high-precision canonicalization after Bronze extraction:

```powershell
python -m medical_kg canonicalize --config config.yaml
```

This mode merges only an unambiguous exact alias, type-compatible abbreviation, or curated
high-precision synonym. Otherwise it creates a new canonical entity, because a false medical merge
is more harmful than a false split. It never modifies `entity_mentions` or `raw_assertions`.

To let the configured LLM decide among retrieved candidates and map relations that remain `OTHER`,
opt in explicitly:

```powershell
python -m medical_kg canonicalize --semantic --config config.yaml
```

The LLM receives the mention type, exact evidence sentence, local context, document title,
candidate names, aliases, types, and available external IDs. A `MATCH` must copy a supplied ID and
meet `canonicalization.confidence_threshold`; otherwise a new entity is retained. Re-running the
command is idempotent. `--document-id` can limit new work while still retrieving candidates from
the full canonical entity index.

## OpenAlex 文献筛选与知识抽取

项目根目录的 `openalex_pipeline.py` 是 OpenAlex 专用核心入口。它流式读取
`data/works/updated_date=*/part_*.gz`，默认按 OpenAlex Field 保留医学与生物医学文献，
并支持标题/摘要关键词、source、全文可用性提示、
分批 LLM 编号筛选、显式 Work ID 追加、正文准备，以及复用现有流水线抽取知识关系。

```powershell
python openalex_pipeline.py run D:\openalex-snapshot `
  --keyword diabetes --keyword voice --keyword-mode all `
  --source "Journal of Voice" --require-fulltext `
  --llm-prompt "筛选研究糖尿病与声音或声学特征关系的原创研究，排除综述。" `
  --llm-batch-size 20 `
  --fulltext-dir D:\openalex-fulltext `
  --content-mode fulltext `
  --workspace data/openalex
```

确认候选文档后，增加 `--extract --config config.yaml` 即可将已准备文档注册到现有
SQLite 知识库并抽取关系。筛选记录保存在 `data/openalex/catalog.sqlite3`，稳定编号采用
`W...` / `openalex:W...`，并保留原始 Work JSON、gzip 分片路径和行号。完整命令、正文
来源约定、增量追加与结果结构见 [OpenAlex 使用说明](docs/openalex_pipeline.md)。

## Architecture

```text
Local papers
  -> documents + processing_jobs
  -> full-document LLM passes
  -> entity_mentions + raw_assertions + evidence + extraction_runs
  -> entity_resolutions + entities + aliases
  -> relation normalization + canonical assertions + assertion_evidence
  -> [Future] embedding/ontology candidate providers and hierarchy enrichment
```

The SQLite file is authoritative. Prompt files live in `prompts/`, runtime configuration in
`config.example.yaml`, and the canonical relation vocabulary in `config/relations.yaml`.

### SQLite raw-output storage

The complete provider response is stored once on its `extraction_runs` row. Individual
`raw_assertions` retain only the structured assertion, evidence, qualifiers, validation state, and
provenance; they do not repeat the complete response. Chunk checkpoints keep their validated and
raw result so interrupted extraction can resume without another API request.

## Development

```bash
conda activate medical-kg
python -m pip install -e ".[dev]"
pytest
ruff check .
```

The initial schema is applied idempotently by `python -m medical_kg init-db`. Before changing a
deployed schema, introduce versioned Alembic migrations under `src/medical_kg/db/migrations/`.

## Running Commands

```bash
python -m medical_kg status --config config.yaml
python -m medical_kg stats --config config.yaml
python -m medical_kg run "data/knowledge_base" --source-type guidelines --chunk-size 12000 --chunk-overlap 500 --config config.yaml
```

## Neo4j 导入与 Web 浏览

`src/utils/sqlite_to_neo4j.py` 会完整保留 SQLite 的表、行、列和外键，并将 Bronze
`raw_assertions` 及 Gold `assertions` 额外投影为可直接浏览的知识关系。Web 页面默认只显示
Gold 规范化图；Bronze 原始图作为独立诊断视图保留，不应据此判断最终 KG 连通性。Gold
边包含聚合后的 `support_count`。先运行 `canonicalize`，再启动本地 Neo4j。
如果 Neo4j 已禁用身份验证，运行：

```powershell
python src/utils/sqlite_to_neo4j.py all --no-auth --clear --open-browser
```

如果 Neo4j 启用了身份验证，则设置密码后运行：

```powershell
$env:NEO4J_PASSWORD = "你的 Neo4j 密码"
python src/utils/sqlite_to_neo4j.py all --clear --open-browser
```

默认读取 `data/medical_kg.sqlite3`，连接 `bolt://localhost:7687`，并在
`http://127.0.0.1:8000` 提供无需额外前端依赖的可视化页面。常用的独立操作如下：

```powershell
# 不连接 Neo4j，只查看将要导入的表和行数
python src/utils/sqlite_to_neo4j.py inspect

# 仅导入（重复执行会按 SQL 主键更新，不会产生重复节点）
python src/utils/sqlite_to_neo4j.py import --clear

# 仅启动 Web 页面
python src/utils/sqlite_to_neo4j.py serve --port 8000
```

连接参数也可用 `--uri`、`--user`、`--password`、`--database` 指定。`--no-auth` 与密码
参数互斥。`--clear` 只删除带 `SQLRow` 标签、即由此脚本导入的节点，不影响 Neo4j 中的
其他数据。
