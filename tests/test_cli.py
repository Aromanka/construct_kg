from pathlib import Path

from medical_kg.cli import build_parser
from medical_kg.config import ProcessingSettings


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
    assert args.progress_style == "bar"

    log_args = build_parser().parse_args(["run", "--progress-style", "log"])
    assert log_args.progress_style == "log"


def test_canonicalize_parser_accepts_silver_options() -> None:
    args = build_parser().parse_args(
        ["canonicalize", "--semantic", "--document-id", "doc-1"]
    )

    assert args.command == "canonicalize"
    assert args.semantic is True
    assert args.document_id == "doc-1"
    assert args.progress_style == "bar"

    log_args = build_parser().parse_args(
        ["canonicalize", "--progress-style", "log"]
    )
    assert log_args.progress_style == "log"


def test_stats_parser_accepts_config() -> None:
    args = build_parser().parse_args(["stats", "--config", "local.yaml"])

    assert args.command == "stats"
    assert args.config == Path("local.yaml")


def test_default_api_concurrency_is_one_hundred() -> None:
    settings = ProcessingSettings()

    assert settings.api_concurrency == 100
    assert settings.job_concurrency == 100


def test_legacy_max_concurrency_is_migrated() -> None:
    settings = ProcessingSettings.model_validate({"max_concurrency": 12})

    assert settings.api_concurrency == 12
    assert settings.job_concurrency == 12
