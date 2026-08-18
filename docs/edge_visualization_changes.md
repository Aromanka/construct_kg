# Edge Visualization Enhancement - Implementation Summary

## Changes Made

Added click-to-view functionality for edges in the Neo4j web visualization (both `sqlite_to_neo4j.py` and `sqlite_gold_to_neo4j.py`).

## Features Implemented

### 1. Edge Click Detection
- Added `nearestEdge(p)` function that:
  - Computes point-to-line-segment distance for all edges
  - Uses threshold of 8/scale pixels for hit detection
  - Returns the closest edge within threshold

### 2. Click Handler Enhancement
- Modified canvas click handler to check both nodes and edges
- Priority: nodes first, then edges
- Displays edge properties in the existing right sidebar

### 3. Edge Property Display
- Shows all Neo4j relationship properties (canonical relation, qualifiers, etc.)
- Highlights important flags:
  - ⚠ **否定关系** (negated relationship) - orange warning
  - ⚠ **推测性关系** (speculative relationship) - orange warning
  - **支持证据数** (support_count) - evidence count in blue box

### 4. Visual Feedback
- Edges highlight on hover (white, thicker stroke)
- Selected edges show in sidebar with all properties
- Search box now finds both nodes and edges

### 5. UI Text Updates
- Hint text changed from "单击任意节点" to "单击节点或关系"

## What's Displayed for Each Edge

**Currently shown (from Neo4j properties):**
- `assertion_id` - UUID linking back to SQLite
- `relation` - canonical relation name (e.g., TREATS, CAUSES, OTHER)
- `canonical_relation_id` - controlled vocabulary ID
- `qualifiers` - JSON object with fact-level metadata
- `negated` - boolean flag
- `speculative` - boolean flag
- `support_count` - number of supporting evidence entries
- `sql_key` - stable identifier

**Not yet shown (requires SQLite join):**
- Original `detailed_relation` phrase from LLM extraction
- Evidence text quotes
- Source document titles/filenames/DOI/PMID
- Per-evidence confidence scores

## Future Enhancement Path

To show full edge provenance (source papers + evidence quotes):

1. Add backend API endpoint `/api/edge-evidence?assertion_id=<uuid>`
2. Query SQLite: `assertions → assertion_evidence → documents → raw_assertions`
3. Add "View Evidence (N)" button in sidebar that fetches and displays:
   - Document title, DOI, PMID per supporting paper
   - Verbatim evidence quote from each paper
   - Original uncanonicalized relation phrase
   - LLM confidence score

## Files Modified

- `src/utils/sqlite_to_neo4j.py` - all changes in embedded HTML/JS
- `src/utils/sqlite_gold_to_neo4j.py` - no changes needed (reuses `shared.serve()`)

## Testing

Run either script to test:
```bash
python src/utils/sqlite_to_neo4j.py all --no-auth --clear --open-browser
# or Gold-only version:
python src/utils/sqlite_gold_to_neo4j.py all --no-auth --clear --open-browser
```

Click on any edge in the graph to see its properties in the right sidebar.
