# Edge Evidence Display - Implementation Summary

## What Was Implemented

Added full edge evidence display to the Neo4j web visualization, showing:
- Original free-text relation phrases from LLM extraction
- Source document titles and filenames
- Evidence text quotes
- Document metadata (DOI, PMID)

## Changes Made

### 1. Backend API (`sqlite_to_neo4j.py`)

**New function: `query_edge_evidence()`** (lines 504-548)
- Queries SQLite for evidence details of a Gold assertion
- Joins `assertion_evidence` → `documents` → `raw_assertions`
- Returns structured JSON with evidence array

**New API endpoint: `/api/edge-evidence`** (lines 574-585)
- Accepts `assertion_id` query parameter
- Returns evidence details or error response
- Integrated into `GraphRequestHandler.do_GET()`

**Updated `serve()` function** (line 604)
- Now accepts optional `sqlite_path` parameter
- Passes SQLite path to request handler for evidence queries

**Updated `query_graph()` function** (line 452)
- Now stores `assertion_id` in edge properties for Gold assertions
- Frontend uses this ID to fetch evidence details

### 2. Frontend JavaScript (embedded in `WEB_PAGE`)

**Enhanced `display()` function** (line 717)
- Detects when an edge is clicked
- Shows loading message while fetching evidence
- Makes async fetch to `/api/edge-evidence` endpoint
- Renders evidence details with proper formatting

**Evidence display includes:**
- Document title (or document_id as fallback)
- File path
- DOI and PMID (if available)
- Original LLM relation phrase (`detailed_relation`)
- Verbatim evidence text
- Confidence score

### 3. Both Scripts Updated

Both `sqlite_to_neo4j.py` and `sqlite_gold_to_neo4j.py` received the same changes:
- `sqlite_gold_to_neo4j.py` calls `shared.serve()` with the SQLite path
- All functionality works for both full DB and Gold-only modes

## Data Flow

```
User clicks edge in graph
  ↓
Frontend extracts assertion_id from edge properties
  ↓
Fetch /api/edge-evidence?assertion_id=...
  ↓
Backend queries SQLite:
  assertion_evidence → documents (file_path, title, DOI, PMID)
  assertion_evidence → raw_assertions (detailed_relation)
  ↓
JSON response with evidence array
  ↓
Frontend renders evidence in sidebar
```

## Database Schema Used

No new tables needed - all data comes from existing tables:
- `assertions` - canonical subject/relation/object facts
- `assertion_evidence` - links assertions to supporting documents + evidence text
- `raw_assertions` - original LLM extraction with `detailed_relation` phrase
- `documents` - source files with title, file_path, DOI, PMID

## Testing

✓ Script syntax validated (both scripts parse correctly)
✓ API function tested (returns expected structure)
✓ No additional database setup required

## How to Use

1. Run the pipeline to populate Gold data:
   ```bash
   python -m medical_kg canonicalize
   ```

2. Launch the web interface:
   ```bash
   python src/utils/sqlite_gold_to_neo4j.py all --no-auth --clear --open-browser
   ```

3. Click any edge in the graph to see:
   - Canonical relation properties
   - Original LLM relation phrase
   - Evidence text quotes
   - Source document details

## Example Output

When clicking an edge, the sidebar now shows:

```
Edge Details
Type: ASSERTION
Relation: TREATS

Properties:
{
  "assertion_id": "abc123...",
  "support_count": 3,
  ...
}

证据详情
来源 1: "Diabetes Treatment Guidelines 2024"
文件: papers/diabetes_guidelines.txt
原始关系短语: "metformin effectively reduces blood glucose levels"
证据文本: "In our study, metformin showed significant reduction..."
置信度: 0.95

来源 2: "Clinical Review of Metformin"
...
```

## Notes

- Evidence loading is asynchronous - shows "加载证据详情..." while fetching
- Gracefully handles missing data (no file_path, no DOI, etc.)
- Error messages displayed in sidebar if API call fails
- Works with both Chinese and English content
- No performance impact - evidence only loaded when edge is clicked
