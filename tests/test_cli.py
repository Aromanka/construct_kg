from pathlib import Path

import pytest

from medical_kg.cli import build_parser, main


def test_run_parser_accepts_lightweight_module_command_options() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "data/knowledge_base",
            "--source-type",
            "guidelines",
            "--chunk-size",
            "12000",
            "--chunk-overlap",
            "500",
        ]
    )

    assert args.source == Path("data/knowledge_base")
    assert args.source_type == "guidelines"
    assert args.chunk_size == 12000
    assert args.chunk_overlap == 500


def test_canonicalize_command_does_not_require_runtime_services(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["canonicalize"]) == 0
    captured = capsys.readouterr()
    assert "intentionally gated" in captured.out


def test_stats_parser_accepts_config() -> None:
    args = build_parser().parse_args(["stats", "--config", "local.yaml"])

    assert args.command == "stats"
    assert args.config == Path("local.yaml")
