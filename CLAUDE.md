# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A Python 3.10+ pipeline that converts local biomedical papers into a SQLite-backed knowledge graph with a Bronze/Silver/Gold architecture. The SQLite file is authoritative — prompt YAML, configuration YAML, and the relation vocabulary are inputs; everything else is derived state.

The pipeline runs full-document LLM extraction passes (`general`, `molecular`, `clinical`), then performs deterministic Silver entity resolution + relation canonicalization into a Gold graph. An optional `OpenAlex` sub-pipeline screens/registers literature at scale. A separate `src/pnet/` module builds path networks over the Gold graph.

## Common commands

Run all commands from the project root with the Conda env active.

Setup:
```bash
conda create -n medical-kg python=3.10 -y
conda activate medical-kg
python -m pip install -e .           # core runtime
python -m pip install -e ".[pdf]"    # only if PDF ingestion is needed
python -m pip install -e ".[dev]"    # adds pytest, pytest-asyncio, ruff
cp config.example.yaml config.yaml   # required before running
```

Initialize schema and seed the relation vocabulary:
```bash
python -m medical_kg init-db --config config.yaml
```

Core pipeline:
```bash
python -m medical_kg ingest ./papers --source-type research
python -m medical_kg extract
python -m medical_kg run ./papers
python -m medical_kg run ./guidelines --source-type guidelines --chunk-size 12000 --chunk-overlap 500
python -m medical_kg run --limit 100
python -m medical_kg run ./papers --progress-style log   # tqdm bars are default
python -m medical_kg extract --document-id PMID:123456
python -m medical_kg retry-failed
python -m medical_kg status
python -m medical_kg stats
python -m medical_kg canonicalize                # deterministic only
python -m medical_kg canonicalize --semantic     # enable LLM tie-breakers
python -m medical_kg canonicalize --rebuild      # discard Silver/Gold, recompute
python -m medical_kg canonicalize --batch-size 500
```

Tests and lint:
```bash
pytest                                   # all tests (asyncio_mode = "auto")
pytest tests/test_canonicalization.py     # single test file
pytest -k entity_resolution               # single test by name
ruff check .
```

OpenAlex screening pipeline (separate entry point that streams gzipped snapshot shards):
```bash
python openalex_pipeline.py run D:\openalex-snapshot \
  --keyword diabetes --keyword voice --keyword-mode all \
  --source "Journal of Voice" --require-fulltext \
  --llm-prompt "筛选..." --llm-batch-size 20 \
  --fulltext-dir D:\openalex-fulltext --content-mode fulltext \
  --workspace data/openalex
python openalex_pipeline.py run <snapshot> --extract --config config.yaml
```

Gold → Neo4j export (the Web page only shows Gold; Bronze is a diagnostic view):
```bash
python src/utils/sqlite_to_neo4j.py all --no-auth --clear --open-browser
python src/utils/sqlite_gold_to_neo4j.py all --sqlite data/medical_kg.sqlite3 --no-auth --clear --open-browser
python src/utils/sqlite_to_neo4j.py inspect
python src/utils/sqlite_to_neo4j.py import --clear
python src/utils/sqlite_to_neo4j.py serve --port 8000
```

Path-network builder over Gold (default algorithm is Bounded Bidirectional Corridor PNet — `BBC-PNet`):
```bash
python src/pnet/build_pnet.py --config src/pnet/build_config.json
python src/pnet/build_pnet.py --config src/pnet/build_config.json --max-layers 9 --max-hops 8 ...
```

PNet → Neo4j visualization (reads TSV files from PNet output directory):
```bash
python src/utils/pnet_to_neo4j.py all --pnet-dir src/pnet/openalex_1 --no-auth --clear --open-browser
python src/utils/pnet_to_neo4j.py inspect --pnet-dir src/pnet/openalex_1
python src/utils/pnet_to_neo4j.py import --pnet-dir src/pnet/openalex_1 --no-auth --clear
python src/utils/pnet_to_neo4j.py serve --no-auth --port 8000
```

## Configuration

- `config.yaml` (git-ignored) — runtime config; supports `${VAR}` / `${VAR:-default}` env expansion in `AppSettings._expand_environment`.
- `config.example.yaml` — committed template.
- `config/relations.yaml` — the controlled relation vocabulary (≈35 names + `OTHER`); loaded and seeded into `relation_types` by `init-db`.
- `prompts/extraction.yaml` — versioned prompts for `general` / `molecular` / `clinical` passes; the registry reads each definition's own `version`. Bump `extraction.stage_version` in config when extraction semantics change; chunk settings are part of that effective version, and prompt version is auto-included so prompt changes never reuse incompatible checkpoints.
- `src/medical_kg/db/migrations/*.sql` — additive DDL applied after `create_all`. Initial schema is generated from SQLAlchemy metadata in `src/medical_kg/db/models.py`; before changing a deployed schema, introduce a new numbered Alembic-style file under that directory.

`DEEPSEEK_API_KEY` must be set (or replaced inline in `config.yaml`). The HTTP client explicitly uses the OS trusted CA context so a stray `SSL_CERT_FILE` is ignored.

## Architecture

Layered Bronze → Silver → Gold, each persisted in SQLite:

- **Landing** (`src/medical_kg/landing/`): `DocumentLoader` ingests `.txt`, `.md`, `.json`, optional `.pdf` into `documents` + immutable `document_revisions`. Stable `document_id` is the relative path or JSON `document_id`. Chunking in `landing/chunking.py` is opt-in via `--chunk-size` / `--chunk-overlap`.
- **Bronze** (`src/medical_kg/bronze/extraction.py`): `BronzeExtractor` runs full-document (or chunked) LLM passes. Each chunk has its own `extraction_chunk_jobs` row with `(document_id, pass_name, stage_version, content_hash, chunk_index)` uniqueness and stores both validated + raw output + provider/model/temperature. A successful `(document_id, pass, effective stage version)` is skipped on subsequent runs. Output → `entity_mentions` + `raw_assertions` + `evidence` + `extraction_runs`. Rate limiting is local via `SmoothRateLimiter` plus a SQLite-backed shared limiter when `processing.distributed_rate_limit: true`. Job claiming is atomic with heartbeats / lease expiration / `RUNNING` recovery in `PipelineRunner.extract` (stale threshold = `2 * (request_timeout * (max_retries + 1) + 60 * max_retries)`).
- **Silver** (`src/medical_kg/silver/`): deterministic entity resolution via `IndexedEntityResolver` (exact normalized alias → type-scoped synonym → order-insensitive token set → abbreviation → curated synonym). Optional `--semantic` adds an LLM tie-breaker that must echo a supplied candidate ID and meet `canonicalization.confidence_threshold`. `ExactRelationNormalizer` maps relation names to the controlled vocabulary; LLM fallback maps `OTHER`. Never merges on lexical similarity alone.
- **Gold** (`src/medical_kg/gold/validation.py` + canonicalization): deduplicates via `silver/deduplication.normalized_assertion_identity` (canonical subject/relation/object + semantic qualifiers + negation + speculation). Repeated facts aggregate into separate `assertion_evidence` rows. Gold assertions carry `support_count` after import.
- **Statistics** (`src/medical_kg/utils/statistics.py`): read-only counters for `bronze_knowledge.raw_assertions` (extracted), `canonical_knowledge.assertions` (deduped facts), `canonical_knowledge.evidence_links` (raw supporting), plus graph-quality metrics (singleton edges, largest CC, canonical compression, cross-document reuse, `OTHER` relation share).

LLM client (`src/medical_kg/llm/`): `CompatibleAPIClient` is a lightweight `httpx`-based DeepSeek / OpenAI-compatible client behind `LLMClient` (no `openai` SDK). It allows `max_request_count = 2` requests per call so it can issue one guided correction request for otherwise invalid structured output; strict Pydantic validation still gates writes (`StructuredOutputValidationError`).

OpenAlex sub-pipeline (`src/medical_kg/openalex/` + `openalex_pipeline.py`): streams `data/works/updated_date=*/part_*.gz`, filters by OpenAlex Field + title/abstract keywords + sources + LLM batch screening + `--include-ids`, resolves fulltext, and (with `--extract`) reuses the main pipeline's extractor. Catalog is `data/openalex/catalog.sqlite3`; document IDs are `W...` / `openalex:W...`.

PNet (`src/pnet/`): builds path networks over Gold. Default algorithm is **Bounded Bidirectional Corridor PNet (BBC-PNet)** — computes `ds`/`dt` distances, keeps real KG corridors with `ds + dt <= max_hops`, then fixed-depth temporal expansion. The older Dual-Keyword Bidirectional BFS Frontier Bridge is selectable via config. See `src/pnet/corridor_algorithm.md`.

Neo4j export (`src/utils/`): `sqlite_to_neo4j.py` preserves all tables/rows/columns/FKs and additionally projects Bronze `raw_assertions` + Gold `assertions` as browsable relations; `sqlite_gold_to_neo4j.py` is a Gold-only path that skips Bronze / document bodies / jobs tables for memory pressure. Both write a `SQLRow` label so `--clear` is scoped. Web page at `http://127.0.0.1:8000` shows Gold by default; Bronze is a separate diagnostic view.

## Concurrency / runtime notes

- WAL mode + `busy_timeout` + `BEGIN IMMEDIATE` are configured in `cli.py`. SQLite has one writer at a time — keep the DB on local disk, not NFS.
- `KnowledgeRepository._write_lock` serializes writers inside a process; `_write_session` opens each transaction with `BEGIN IMMEDIATE`.
- `processing.heartbeat_interval` must be smaller than `processing.job_lease_seconds` (validated in config).
- Concurrency caps (`job_concurrency`, `api_concurrency`, `chunk_queue_size`) are upper bounds; actual in-flight requests are gated by provider RPM/TPM and shared SQLite rate-limit state.
- `canonicalize` is idempotent and resumable: each committed batch is a durable checkpoint; re-running skips already-resolved mentions / already-materialized evidence. `--rebuild` discards all derived Silver/Gold state but preserves Bronze extraction rows.

## Input contract for documents

Plain text / Markdown use relative path as `document_id`. JSON files may be either a string or `{"document_id", "title", "doi", "pmid", "source_type", "content"}`. Changing content under an existing `document_id` resets its processing jobs but keeps prior Bronze extraction rows + the exact source revision for provenance.

## Tests

`tests/` mirrors the architecture (`test_pipeline_resume`, `test_repository`, `test_extraction_schema`, `test_canonicalization`, `test_entity_resolution`, `test_relation_normalization`, `test_sources_and_chunking`, `test_statistics`, `test_compatible_llm_client`, `test_cli`, `test_openalex_pipeline`, `test_sqlite_gold_to_neo4j`, `test_pnet_builder`, `test_python_compatibility`). `asyncio_mode = "auto"` — async tests need no decorator.