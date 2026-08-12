"""List diabetes disease candidates using projected disease-node columns."""

from __future__ import annotations

import argparse
from pathlib import Path

from optimus_diabetes import (
    DEFAULT_KG_ROOT,
    DEFAULT_QUERY,
    DEFAULT_ROOT_NAME,
    choose_root,
    find_disease_candidates,
    validate_kg_root,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-root", type=Path, default=DEFAULT_KG_ROOT)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--root-name", default=DEFAULT_ROOT_NAME)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    root = validate_kg_root(args.kg_root, edge_types=[])
    candidates = find_disease_candidates(root, args.query, args.root_name)
    selected = choose_root(candidates)
    print(f"Selected root: {selected['id']} | {selected['name']}")
    print(f"Matched candidates: {len(candidates)}\n")
    for row in candidates[: max(0, args.limit)]:
        print(
            f"{row['selection_score']:4d}  {row['id']:20s}  "
            f"{row['name']}  [{row['matched_fields']}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
