"""Print the summary produced by extract_diabetes_subgraph.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimus_diabetes import DEFAULT_OUTPUT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary_path = args.output_dir.expanduser().resolve() / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Summary not found: {summary_path}. Run extract_diabetes_subgraph.py first."
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        print(json.dumps(json.load(handle), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
