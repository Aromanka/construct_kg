"""Backward-compatible alias for 01_find_diabetes.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("01_find_diabetes.py")), run_name="__main__")
