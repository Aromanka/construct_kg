# OpenAlex Pipeline Architecture

## Overview

The OpenAlex pipeline is a specialized sub-system within the medical knowledge graph project that enables large-scale literature screening, selection, and knowledge extraction from OpenAlex snapshot datasets. It operates independently from the main pipeline but integrates seamlessly with the core extraction and knowledge graph construction workflows.

**Purpose**: Screen millions of biomedical papers from OpenAlex snapshots, select relevant works based on configurable filters and LLM-based relevance screening, materialize full-text or abstract content, and optionally feed selected documents into the main knowledge extraction pipeline.

**Entry Point**: `openalex_pipeline.py` → delegates to `src/medical_kg/openalex/cli.py`

## Architecture Components

### 1. Snapshot Reader (`src/medical_kg/openalex/snapshot.py`)

**Responsibility**: Stream compressed OpenAlex snapshot data without full materialization.

**Key Features**:
- Reads gzipped JSONL files (`part_*.gz`) from OpenAlex snapshot directory structure
- Supports manifest-based ordering (reads `manifest` file if present, falls back to sorted discovery)
- Graceful error handling: corrupt/truncated parts are logged and skipped (strict mode aborts instead)
- Progress tracking via byte-level callbacks for tqdm integration
- Zero-copy streaming: never loads entire gzip files into memory

**Data Flow**:
```
OpenAlex snapshot root
└── data/works/updated_date=*/part_*.gz
    ├── iter_raw() → (dict, file_path, line_number)
    └── iter_works() → OpenAlexWork objects
```

**Error Recovery**:
- `SnapshotReadFailure` records track entity, path, error type, message, and records successfully read before failure
- `strict=False` (default): logs failures and continues
- `strict=True`: raises exception on first corrupt part

---

### 2. Work Models (`src/medical_kg/openalex/models.py`)

**Core Data Structure**: `OpenAlexWork`

**Fields**:
- `work_id`: Normalized short ID (e.g., `W2741809807`)
- `openalex_id`: Full URL (e.g., `https://openalex.org/W2741809807`)
- `document_id`: Stable identifier for KG ingestion (`openalex:W{id}`)
- `title`, `abstract`: Restored from inverted index
- `doi`, `publication_year`, `language`, `work_type`
- `sources`: List of publication venues with IDs and names
- `primary_source_id`, `primary_source_name`
- `has_fulltext_hint`, `fulltext_urls`: Content availability indicators
- `field_ids`: OpenAlex Field taxonomy IDs (medicine, biomedical, etc.)
- `raw`: Complete original JSON for provenance
- `snapshot_file`, `snapshot_line`: Exact source locator

**Utilities**:
- `normalize_work_id()`: Extracts stable short ID from various formats
- `restore_abstract()`: Reconstructs text from OpenAlex inverted index
- `collect_field_ids()`: Extracts Field IDs from topics/primary_topic
- `_content_hints()`: Aggregates fulltext availability signals

---

### 3. Filtering System (`src/medical_kg/openalex/filtering.py`)

**Responsibility**: Deterministic rule-based Work filtering before optional LLM screening.

**Predefined Field Sets**:
- `MEDICAL_CORE_FIELDS`: {27: Medicine, 36: Health Professions}
- `BIOMEDICAL_FIELDS`: {13: Biochemistry/Genetics/Molecular Biology, 24: Immunology/Microbiology, 28: Neuroscience, 30: Pharmacology/Toxicology/Pharmaceutics}
- `MEDICAL_BROAD_FIELDS`: Union of above (default gate)

**WorkFilter Configuration**:
```python
@dataclass(frozen=True)
class WorkFilter:
    allowed_field_ids: tuple[int, ...] | None  # None = no Field gate
    keywords: list[str]                        # Title/abstract keywords
    keyword_mode: str                          # "any" or "all"
    exclude_keywords: list[str]
    sources: list[str]                         # Source ID/name/type selectors
    require_abstract: bool
    require_fulltext: bool
```

**Matching Logic**:
1. Field gate: Work must have at least one Field ID in `allowed_field_ids`
2. Keywords: NFKC-normalized casefold search in title + abstract
   - `keyword_mode="any"`: At least one keyword matches
   - `keyword_mode="all"`: All keywords must match
3. Exclude keywords: Reject if any exclusion term found
4. Sources: JSON-serialized sources must contain at least one selector
5. Abstract requirement: Non-empty restored abstract
6. Fulltext requirement: `has_fulltext_hint=True`

**Returns**: `(matched: bool, matched_keywords: list[str])`

---

### 4. LLM Screening (`src/medical_kg/openalex/screening.py`)

**Responsibility**: Batch relevance screening using the configured LLM API.

**Protocol**: `WorkScreener`
```python
async def screen(works: Sequence[OpenAlexWork], instruction: str) -> set[int]
```

**CompatibleWorkScreener Implementation**:
- Uses OpenAI-compatible API (configured via `config.yaml` LLM settings)
- Batches titles + abstracts with 1-based indexing
- System prompt: instructs model to return `{"selected_numbers": [1, 2, ...]}`
- Response format: `{"type": "json_object"}`
- Validation: ensures returned numbers are valid integers within batch range
- Returns: zero-based indexes of selected works

**Batch Format**:
```
[1]
TITLE: ...
ABSTRACT: ...

[2]
TITLE: ...
ABSTRACT: [NO ABSTRACT]
```

---

### 5. Catalog (`src/medical_kg/openalex/catalog.py`)

**Responsibility**: Durable SQLite-backed registry of screened Works.

**Schema** (`catalog.sqlite3`):

**`works` table**:
- Primary key: `work_id` (stable short ID)
- Content: title, abstract, DOI, publication metadata
- Sources: `sources_json`, `primary_source_id/name`
- Selection: `selected`, `selection_method` (`filter`/`llm`/`explicit`/`manual`), `matched_keywords_json`
- Fulltext tracking: `fulltext_path`, `fulltext_status`, `fulltext_materialized_path`, `fulltext_processed`
- Abstract tracking: `abstract_materialized_path`, `abstract_processed`
- Provenance: `snapshot_file`, `snapshot_line`, `raw_json`

**`sources` table**:
- Source registry with `is_full_record` flag (enrichment support)

**`selection_runs` table**:
- Audit log: `started_at`, `options_json`, `scanned`, `candidates`, `selected`

**Key Operations**:
- `upsert_work()`: Insert or update, preserving `manual` selection method
- `selected_rows()`: Query selected works with optional limit
- `set_fulltext_materialized()`: Record fulltext resolution status
- `set_abstract_materialized()`: Record abstract batch membership
- `selected_materialized_paths()`: Return paths to materialized documents by content mode

**Forward Compatibility**: Automatic schema migration adds missing columns from newer releases.

---

### 6. Full-Text Resolution (`src/medical_kg/openalex/fulltext.py`)

**Responsibility**: Acquire full-text content from local directories or OpenAlex content service.

**Resolution Strategy** (in order):
1. **Local directory lookup**: Check `--fulltext-dir` for `W{id}.{ext}` or `W{id}/W{id}.{ext}`
   - Supported formats: `.grobid-xml`, `.xml`, `.xml.gz`, `.txt`, `.md`, `.json`, `.pdf`
2. **OpenAlex content service** (if `--download-fulltext`):
   - Tries `https://content.openalex.org/works/{id}.grobid-xml`
   - Tries `https://content.openalex.org/works/{id}.pdf`
   - Tries any `content_urls` from Work metadata pointing to `content.openalex.org`
   - Requires `--openalex-api-key` (or `OPENALEX_API_KEY` env var) for authorized access

**Status Codes**:
- `local`: Found in local directory
- `downloaded`: Downloaded from OpenAlex
- `not_found`: No source available
- `download_failed`: HTTP errors or parsing failures
- `quota_unavailable`: HTTP 402/429 (quota exceeded)
- `unauthorized`: HTTP 401/403

**Content Parsing**:
- XML/GROBID: Extract all text nodes
- TXT/MD: UTF-8 read
- JSON: Extract from `full_text`, `content`, or `text` field
- PDF: Requires `medical-kg[pdf]` (pypdf)

**Circuit Breaker**: After first 401/402/403/429, disables remote downloads for session.

---

### 7. Pipeline Orchestrator (`src/medical_kg/openalex/pipeline.py`)

**Responsibility**: Coordinate snapshot streaming, filtering, screening, cataloging, and materialization.

**Core Workflow**:

#### **Selection Phase** (`select()`)
```
1. Start selection run in catalog (log options)
2. Stream Works from snapshot
3. For each Work:
   a. Check if explicitly included (--include-id)
   b. Apply WorkFilter
   c. Accumulate candidates in batch
   d. When batch full:
      - Optional: LLM screening
      - Upsert to catalog with selection status
4. Finish selection run (log statistics)
```

**Stopping Conditions**:
- `max_works`: Total Works scanned from snapshot
- `max_candidates`: Works admitted by filter (before LLM)
- `max_selected`: Works marked selected (after LLM)

**Selection Methods**:
- `explicit`: Passed via `--include-id`
- `filter`: Admitted by WorkFilter
- `llm`: Admitted by LLM screener
- `manual`: Marked via `mark_manual()` (sticky across runs)

#### **Materialization Phase** (`materialize()`)
```
1. Query selected Works from catalog
2. For each selected Work:
   a. If content_mode="fulltext":
      - Resolve fulltext via FullTextResolver
      - Write {work_id}.json with full_text field
      - Update catalog: fulltext_processed=1
   b. If content_mode="abstract":
      - Accumulate abstracts into batches
      - Write abstract_batch_{digest}.json
      - Update catalog: abstract_processed=1
   c. If content_mode="fulltext-or-abstract":
      - Try fulltext first
      - Fall back to abstract if download_failed
```

**Content Modes**:
- `fulltext`: Only resolve full-text (skip if unavailable)
- `abstract`: Stack abstracts into character-bounded batches
- `fulltext-or-abstract`: Fulltext preferred, abstract fallback

**Abstract Batching**:
- Chunk size: `--abstract-chunk-size` (default 12000 characters)
- Format: `[OPENALEX WORK {id}]\nTitle: ...\nDOI: ...\nAbstract: ...\n[/OPENALEX WORK {id}]`
- Batch digest: SHA256 hash (first 24 chars) of concatenated content
- Document ID: `openalex:abstract-batch:{digest}`
- Preserves article boundaries (single long abstract kept intact for downstream chunking)

**Output Structure**:
```
workspace/
├── catalog.sqlite3
├── fulltext/
│   ├── W123.grobid-xml
│   └── W456.pdf
└── documents/
    ├── W123.json
    ├── W456.json
    └── abstract_batches/
        └── abstract_batch_{digest}.json
```

---

### 8. CLI Interface (`src/medical_kg/openalex/cli.py`)

**Commands**:

#### `select`
Screen snapshot Works into durable catalog (no materialization).

```bash
python openalex_pipeline.py select <snapshot> \
  --workspace data/openalex \
  --keyword diabetes --keyword voice --keyword-mode all \
  --source "Journal of Voice" \
  --require-fulltext \
  --llm-prompt "筛选..." --llm-batch-size 20 \
  --max-selected 1000
```

#### `run`
Complete pipeline: select + materialize + optionally extract.

```bash
python openalex_pipeline.py run <snapshot> \
  --keyword diabetes --keyword-mode any \
  --llm-prompt @prompts/relevance.txt \
  --fulltext-dir D:\openalex-fulltext \
  --content-mode fulltext-or-abstract \
  --extract \
  --config config.yaml
```

#### `add`
Register explicit Work IDs from snapshot (bypasses filters).

```bash
python openalex_pipeline.py add <snapshot> W123 W456 W789 \
  --workspace data/openalex
```

#### `materialize`
Materialize previously selected Works (re-runnable).

```bash
python openalex_pipeline.py materialize <snapshot> \
  --workspace data/openalex \
  --fulltext-dir D:\fulltext \
  --content-mode fulltext
```

#### `show`
Display catalog entry for a Work ID.

```bash
python openalex_pipeline.py show W2741809807 \
  --workspace data/openalex
```

**Filter Options**:
- `--field ID`: OpenAlex Field ID (repeatable; replaces default medical gate)
- `--no-medical-field-filter`: Disable Field gate
- `--keyword TEXT`: Title/abstract keyword (repeatable)
- `--keyword-mode any|all`
- `--exclude-keyword TEXT`
- `--source TEXT`: Source ID/name/type (repeatable)
- `--require-abstract`, `--require-fulltext`
- `--llm-prompt TEXT`: Relevance instruction (or `@path` to read file)
- `--llm-batch-size N`: Screening batch size (default 20)
- `--include-id W{id}`: Always select (repeatable)
- `--all`: Select all Works (requires explicit flag when no filters)

**Materialization Options**:
- `--fulltext-dir PATH`: Local fulltext directory
- `--download-fulltext` / `--no-download-fulltext`
- `--openalex-api-key KEY`
- `--content-mode fulltext|abstract|fulltext-or-abstract`
- `--abstract-chunk-size N`: Max characters per abstract batch (default 12000)

**Extraction Integration**:
- `--extract`: Ingest materialized documents and run Bronze extraction
- `--chunk-size`, `--chunk-overlap`: Chunking parameters
- `--config PATH`: Configuration file for LLM client and extraction settings

**Safety**:
- `--strict-snapshot`: Abort on corrupt snapshot part (default: log and skip)

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ OpenAlex Snapshot (gzipped JSONL shards)                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓ iter_works()
          ┌────────────────┐
          │ OpenAlexWork   │
          └────────┬───────┘
                   │
                   ↓ WorkFilter.match()
          ┌────────────────┐
          │ Candidates     │
          └────────┬───────┘
                   │
                   ↓ (optional) CompatibleWorkScreener.screen()
          ┌────────────────┐
          │ Selected Works │
          └────────┬───────┘
                   │
                   ↓ upsert_work()
     ┌─────────────────────────┐
     │ OpenAlexCatalog (SQLite) │
     └─────────────┬────────────┘
                   │
                   ↓ materialize()
          ┌────────────────┐
          │ FullTextResolver│
          └────────┬───────┘
                   │
                   ├──→ Local directory
                   │
                   ├──→ content.openalex.org
                   │
                   ↓ Write documents
     ┌──────────────────────────┐
     │ workspace/documents/*.json│
     └──────────┬────────────────┘
                │
                ↓ (optional --extract)
     ┌──────────────────────────┐
     │ Main Pipeline Ingestion  │
     │ → Bronze Extraction      │
     │ → Silver Canonicalization│
     │ → Gold Assertions        │
     └──────────────────────────┘
```

---

## Configuration

**Required Settings** (`config.yaml`):
```yaml
llm:
  base_url: https://api.deepseek.com
  api_key: ${DEEPSEEK_API_KEY}
  model: deepseek-chat
  temperature: 0
  timeout: 120.0
```

**Environment Variables**:
- `DEEPSEEK_API_KEY`: LLM API key for screening and extraction
- `OPENALEX_API_KEY`: OpenAlex content service authorization

---

## Key Design Decisions

### 1. **Why a Separate Catalog?**
- OpenAlex snapshots are 100GB+ compressed; rescanning is expensive
- Catalog persists selection decisions across runs
- Enables iterative refinement: screen once, materialize multiple times with different modes
- Tracks materialization status to avoid redundant downloads

### 2. **Why Batch Abstracts?**
- Abstract-only extraction is orders of magnitude cheaper than fulltext
- Stacking abstracts into 12KB chunks maximizes LLM context utilization
- Document boundaries preserved for provenance (each abstract tagged with Work ID)

### 3. **Why Separate from Main Pipeline?**
- OpenAlex screening operates at snapshot scale (millions of works)
- Main pipeline operates at ingested-document scale (hundreds/thousands)
- Screening uses different rate limits, error handling, and progress tracking
- Clean separation allows independent evolution

### 4. **Manual Selection Method**
- `manual` selection method is sticky: preserved across re-screening runs
- Enables "pin this Work" workflow independent of filter evolution
- Critical for curated datasets

### 5. **Content Mode Fallback**
- `fulltext-or-abstract` mode prevents data loss when quota exhausted
- Materializes abstract batches for Works where fulltext download failed
- Balances completeness vs. quality

---

## Statistics Tracking

**SelectionStatistics**:
- `scanned`: Total Works seen in snapshot
- `candidates`: Works admitted by filter
- `selected`: Works selected (filter + LLM screening)
- `explicit_selected`: Works from `--include-id`
- `selected_with_abstract`, `selected_without_abstract`
- `selected_with_fulltext_hint`, `selected_without_fulltext_hint`
- `selected_with_abstract_and_fulltext_hint`
- `snapshot_parts_failed`: Corrupt parts encountered
- `snapshot_failures`: Detailed failure records

**MaterializeStatistics**:
- `selected`: Works queried from catalog
- `written`: Documents successfully written
- `fulltext`: Fulltext documents
- `abstract_fallback`: Fulltext→abstract fallback (fulltext-or-abstract mode)
- `abstracts`: Individual abstracts in batches
- `abstract_batches`: Batch documents written
- `skipped_processed`: Already materialized
- `download_failed`: HTTP/parsing errors
- `skipped_no_content`: No abstract or fulltext available

---

## Error Handling

### Snapshot Failures
- Corrupt gzip: Log + skip (or abort if `--strict-snapshot`)
- Invalid JSON: Log + skip line
- Invalid Work ID: Log warning, continue

### Download Failures
- HTTP errors: Log warning, mark `download_failed`
- 401/403: Mark `unauthorized`, disable remote for session
- 402/429: Mark `quota_unavailable`, disable remote for session
- Parsing errors: Log warning, try next URL

### LLM Screening Failures
- Invalid response: Raise ValueError (batch fails, no partial selection)
- Out-of-range numbers: Raise ValueError

**Recovery**: All catalog writes use transactions; partial batches never committed.

---

## Integration Points

### With Main Pipeline
1. Materialized documents written as JSON with `document_id`, `content`, `source_type`
2. `--extract` flag invokes `DocumentLoader.ingest()` + `PipelineRunner.extract()`
3. Document IDs: `openalex:W{id}` for fulltext, `openalex:abstract-batch:{digest}` for batches

### With Neo4j Export
- Gold graph visualization includes OpenAlex-sourced assertions
- Document nodes link back to `openalex_work_id` for traceability

### With PNet Builder
- Gold assertions from OpenAlex documents participate in path network construction
- No special handling required (standard entity/relation graph)

---

## Performance Characteristics

**Snapshot Streaming**:
- ~10-50 MB/s decompression throughput (CPU-bound)
- ~100k Works/minute filtering speed (Field + keyword checks)

**LLM Screening**:
- Batch size 20: ~2-5 seconds/batch (model-dependent)
- Rate limited by configured LLM RPM/TPM

**Fulltext Download**:
- ~1-10 documents/second (network-bound)
- Parallelization limited by httpx connection pooling

**Abstract Batching**:
- Instantaneous (deterministic string concatenation)

**Catalog Operations**:
- SQLite WAL mode: concurrent readers, serialized writers
- Batch commits every `llm_batch_size` works (default 20)

---

## Usage Examples

### Basic Keyword Screening
```bash
python openalex_pipeline.py select D:\openalex-snapshot \
  --keyword diabetes --keyword "type 2" --keyword-mode all \
  --require-abstract \
  --max-selected 1000
```

### LLM-Based Relevance Screening
```bash
python openalex_pipeline.py run D:\openalex-snapshot \
  --keyword voice --keyword dysphonia \
  --llm-prompt "Select papers studying voice disorders in diabetic patients" \
  --llm-batch-size 20 \
  --content-mode fulltext-or-abstract \
  --fulltext-dir D:\fulltext \
  --download-fulltext
```

### Extract Knowledge from Selected Works
```bash
python openalex_pipeline.py run D:\openalex-snapshot \
  --workspace data/openalex \
  --include-id W123 W456 W789 \
  --content-mode fulltext \
  --extract \
  --config config.yaml
```

### Materialize Previously Selected Works
```bash
python openalex_pipeline.py materialize D:\openalex-snapshot \
  --workspace data/openalex \
  --content-mode abstract \
  --abstract-chunk-size 15000
```

---

## Future Extensions

**Potential Enhancements**:
1. **Incremental snapshot updates**: Delta processing for `updated_date=*` partitions
2. **Distributed screening**: Multi-machine catalog sharing via network SQLite
3. **Citation graph filtering**: Select works citing/cited-by seed papers
4. **Author/institution filters**: OpenAlex author/institution ID predicates
5. **Concepts/topics filters**: Deprecated Concepts API replacement with topics
6. **PDF OCR fallback**: pypdfium2 for scan-only PDFs
7. **S3/blob storage**: Direct snapshot streaming from cloud storage

---

## Maintenance Notes

**Schema Evolution**:
- `OpenAlexCatalog._create_schema()` auto-migrates missing columns
- Add new columns to `migrations` dict with backward-compatible defaults

**Breaking Changes**:
- Work ID normalization: Always uppercase `W{digits}`
- Abstract restoration: Assumes inverted index schema unchanged
- Content URLs: Hardcoded `content.openalex.org` hostname

**Dependencies**:
- `httpx`: HTTP client for downloads and LLM screening
- `tqdm`: Progress bars
- `pypdf` (optional): PDF text extraction via `medical-kg[pdf]`

---

## Related Documentation

- Main pipeline: See `CLAUDE.md` project overview
- Configuration: `config.example.yaml`
- Extraction prompts: `prompts/extraction.yaml`
- Neo4j export: `src/utils/sqlite_gold_to_neo4j.py`
- PNet builder: `src/pnet/build_pnet.py`
