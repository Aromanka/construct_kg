# Visualization Utilities

This directory contains scripts for importing knowledge graphs and path networks into Neo4j for interactive visualization.

## Gold Knowledge Graph Visualization

Import the canonical Gold graph from SQLite into Neo4j:

```bash
# Import and visualize (default: data/medical_kg.sqlite3)
python src/utils/sqlite_gold_to_neo4j.py all --sqlite data/medical_kg.sqlite3 --no-auth --clear --open-browser

# Inspect Gold tables without importing
python src/utils/sqlite_gold_to_neo4j.py inspect --sqlite data/medical_kg.sqlite3

# Import only (no web server)
python src/utils/sqlite_gold_to_neo4j.py import --sqlite data/medical_kg.sqlite3 --no-auth --clear

# Serve only (assumes already imported)
python src/utils/sqlite_gold_to_neo4j.py serve --no-auth --port 8000
```

**Options:**
- `--clear`: Delete all previous SQLRow nodes (use for fresh import)
- `--no-auth`: Connect to local Neo4j without authentication
- `--password` or `NEO4J_PASSWORD`: Provide password for authenticated Neo4j
- `--batch-size`: Number of rows per batch (default: 500)
- `--delete-batch-size`: Batch size for node deletion (default: 5000)

The web interface displays the canonical Gold graph with:
- Entity nodes with their display names and types
- ASSERTION relationships with relation types and support counts
- Interactive force-directed layout with zoom and pan

## PNet Visualization

Import path networks built by `src/pnet/build_pnet.py`:

```bash
# Import and visualize (default: src/pnet/pnet_output)
python src/utils/pnet_to_neo4j.py all --pnet-dir src/pnet/openalex_1 --no-auth --clear --open-browser

# Inspect PNet without importing
python src/utils/pnet_to_neo4j.py inspect --pnet-dir src/pnet/openalex_1

# Import only
python src/utils/pnet_to_neo4j.py import --pnet-dir src/pnet/openalex_1 --no-auth --clear

# Serve only
python src/utils/pnet_to_neo4j.py serve --no-auth --port 8000
```

**Input files** (must exist in `--pnet-dir`):
- `nodes.tsv`: PNet nodes with layers and entity IDs
- `edges.tsv`: Directed edges between nodes
- `graph.yaml` (optional): Algorithm metadata

The web interface displays:
- Layered layout with nodes arranged by their layer
- Color-coded structural vs knowledge edges
- Node labels showing display names
- Interactive pan and zoom

## Full Database Visualization

For complete SQLite inspection including Bronze/Silver tables and all foreign keys:

```bash
python src/utils/sqlite_to_neo4j.py all --sqlite data/medical_kg.sqlite3 --no-auth --clear --open-browser
```

This imports **all** tables and creates both SQL_FOREIGN_KEY edges and projection edges (RAW_ASSERTION, ASSERTION). Use for debugging and schema exploration.

## Prerequisites

```bash
python -m pip install 'neo4j>=5,<7'
```

For PNet inspection, also install:
```bash
python -m pip install pyyaml
```

## Neo4j Setup

Local Neo4j with authentication disabled:

```bash
# In neo4j.conf
dbms.security.auth_enabled=false
```

Or use `--password` / `NEO4J_PASSWORD` for authenticated instances.

Default connection: `bolt://localhost:7687`
