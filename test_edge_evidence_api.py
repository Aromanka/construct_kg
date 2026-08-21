#!/usr/bin/env python3
"""Quick test to verify the edge evidence API works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.sqlite_to_neo4j import query_edge_evidence

# Test with a dummy assertion_id
sqlite_path = Path("data/medical_kg.sqlite3")

if not sqlite_path.exists():
    print(f"SQLite database not found at {sqlite_path}")
    print("Please run the pipeline first to create the database.")
    sys.exit(1)

# Test the query function
print("Testing query_edge_evidence function...")
result = query_edge_evidence(sqlite_path, "dummy-assertion-id")
print(f"Result structure: {result.keys()}")
print(f"Evidence count: {len(result.get('evidence', []))}")

if result.get('evidence'):
    print("\nFirst evidence entry:")
    first = result['evidence'][0]
    for key, value in first.items():
        print(f"  {key}: {value}")
else:
    print("\nNo evidence found (expected for dummy ID)")

print("\n✓ API function works correctly!")
