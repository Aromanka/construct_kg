"""Core OpenAlex entry point. Run ``python openalex_pipeline.py --help``."""

import sys
from importlib import import_module
from pathlib import Path

# Keep the checked-out repository directly runnable without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
main = import_module("medical_kg.openalex.cli").main


if __name__ == "__main__":
    raise SystemExit(main())
