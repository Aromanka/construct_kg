import ast
from pathlib import Path


def test_source_syntax_is_compatible_with_python_310() -> None:
    source_root = Path(__file__).parents[1] / "src"
    for path in source_root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
