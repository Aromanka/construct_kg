"""Inspect only metadata and a tiny sample; never count/collect entire files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from optimus_diabetes import DEFAULT_KG_ROOT, EDGE_TYPES, validate_kg_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-root", type=Path, default=DEFAULT_KG_ROOT)
    parser.add_argument("--sample-rows", type=int, default=2)
    args = parser.parse_args()
    root = validate_kg_root(args.kg_root)

    files = [root / "nodes" / "disease.parquet"] + [
        root / "edges" / f"{name}.parquet" for name in EDGE_TYPES
    ]
    for path in files:
        parquet = pq.ParquetFile(path)
        print(f"\n=== {path.relative_to(root)} ===")
        print(f"rows={parquet.metadata.num_rows:,}, row_groups={parquet.num_row_groups}")
        print(parquet.schema_arrow)
        if args.sample_rows > 0:
            print(parquet.read_row_group(0).slice(0, args.sample_rows).to_pylist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
