"""Select and display the single disease seed used by the one-hop demo."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--root-id")
    args = parser.parse_args()
    root = validate_kg_root(args.kg_root, edge_types=[])
    seed = choose_root(
        find_disease_candidates(root, args.query, args.root_name), args.root_id
    )
    print(json.dumps({"seed": seed, "ontology_depth": 0}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
