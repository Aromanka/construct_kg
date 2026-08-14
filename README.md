# Medical Literature Knowledge Graph

A maintainable, evidence-aware Python pipeline that converts local biomedical papers into a
resumable SQLite knowledge base. The current implementation deliberately focuses on the
Phase I Bronze pipeline: reliable full-document extraction and faithful persistence of entity
mentions, raw assertions, qualifiers, evidence, model output, and provenance.

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
- SQLite persistence for all Landing, Bronze, Silver, and Gold objects described by the
  architecture specification.
- Atomic job claiming with process-local asynchronous write serialization and SQLite immediate
  transactions, independent failures, bounded
  concurrency, request/token rate limiting, bounded retries, and resume/retry commands.
- Exact source-evidence validation and deterministic mention offsets where possible.
- Structured logs and run-level request/token/success statistics.
- Conservative Silver/Gold extension points. Canonicalization is intentionally gated until the
  Bronze output has been evaluated; it never overwrites Bronze data.

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
of that effective version. Increase `extraction.stage_version` when other extraction semantics
change.

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
extraction passes, entity mentions, raw assertions, evidence validation, and canonical graph
counts:

```powershell
python -m medical_kg stats --config config.yaml
```

For the current Phase I pipeline, `bronze_knowledge.raw_assertions` is the main count of extracted
knowledge. `canonical_knowledge.entities` and `canonical_knowledge.assertions` can remain zero
until canonicalization is enabled. A job left as `RUNNING` after interruption is shown as-is; this
inspection command never changes or resumes it. Only completed extraction passes are committed, so
an interrupted in-flight pass is not included in the knowledge counts.

## Architecture

```text
Local papers
  -> documents + processing_jobs
  -> full-document LLM passes
  -> entity_mentions + raw_assertions + evidence + extraction_runs
  -> [Phase II] entities + relation_types + canonical assertions
  -> [Phase III] ontology enrichment and graph/export projections
```

The SQLite file is authoritative. Prompt files live in `prompts/`, runtime configuration in
`config.example.yaml`, and the canonical relation vocabulary in `config/relations.yaml`.

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
