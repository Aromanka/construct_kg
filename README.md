# Medical Literature Knowledge Graph

A maintainable, evidence-aware Python pipeline that converts local biomedical papers into a
resumable PostgreSQL knowledge base. The current implementation deliberately focuses on the
Phase I Bronze pipeline: reliable full-document extraction and faithful persistence of entity
mentions, raw assertions, qualifiers, evidence, model output, and provenance.

## What is implemented

- Stable document registration using relative paths (or a JSON-provided `document_id`) and
  SHA-256 content hashes, with immutable revision snapshots for historical provenance.
- Text, Markdown, JSON and optional PDF input.
- Full-document `general`, `molecular`, and `clinical` extraction passes.
- Versioned YAML prompts and configurable relation vocabulary.
- OpenAI and OpenAI-compatible structured-output clients behind an `LLMClient` interface.
- Pydantic validation, including controlled biomedical entity types, `OTHER` detail, extensible
  qualifiers, and confidence bounds.
- PostgreSQL persistence for all Landing, Bronze, Silver, and Gold objects described by the
  architecture specification.
- Transactional job claiming with `FOR UPDATE SKIP LOCKED`, independent failures, bounded
  concurrency, request/token rate limiting, bounded retries, and resume/retry commands.
- Exact source-evidence validation and deterministic mention offsets where possible.
- Structured logs and run-level request/token/success statistics.
- Conservative Silver/Gold extension points. Canonicalization is intentionally gated until the
  Bronze output has been evaluated; it never overwrites Bronze data.

## Setup

Python 3.11+ and PostgreSQL are required.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
copy config.example.yaml config.yaml
```

Set `OPENAI_API_KEY` and `POSTGRES_PASSWORD`, or replace their environment placeholders in the
local `config.yaml`. That file is ignored by Git. For PDFs, install `pip install -e ".[pdf]"`.

Create the schema and seed the configured relation vocabulary:

```bash
medical-kg init-db
```

## Input

Plain text and Markdown files use their path relative to the ingested directory as the stable
document ID. A JSON file can contain either a string or this object:

```json
{
  "document_id": "PMID:123456",
  "title": "Example study",
  "doi": "10.0000/example",
  "pmid": "123456",
  "content": "Complete paper text..."
}
```

Changing the content under an existing `document_id` resets its processing jobs. Previous Bronze
extraction records and their exact source revision remain available for provenance.

## Commands

```bash
medical-kg ingest ./papers
medical-kg extract
medical-kg run ./papers
medical-kg run --limit 100
medical-kg extract --document-id PMID:123456
medical-kg retry-failed
medical-kg status
medical-kg canonicalize
```

Every command accepts `--config PATH` where applicable. `run` optionally ingests a directory and
then extracts pending work. A successful `(document_id, extraction pass, stage version)` is skipped
on subsequent runs. Increase `extraction.stage_version` when extraction semantics change.

## Architecture

```text
Local papers
  -> documents + processing_jobs
  -> full-document LLM passes
  -> entity_mentions + raw_assertions + evidence + extraction_runs
  -> [Phase II] entities + relation_types + canonical assertions
  -> [Phase III] ontology enrichment and graph/export projections
```

PostgreSQL is authoritative. Prompt files live in `prompts/`, runtime configuration in
`config.example.yaml`, and the canonical relation vocabulary in `config/relations.yaml`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

The initial schema is applied idempotently by `medical-kg init-db`. Before changing a deployed
schema, introduce versioned Alembic migrations under `src/medical_kg/db/migrations/`.
