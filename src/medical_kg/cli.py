from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from medical_kg.bronze.extraction import BronzeExtractor
from medical_kg.config import AppSettings, load_settings
from medical_kg.db.repository import KnowledgeRepository
from medical_kg.landing.loader import DocumentLoader
from medical_kg.llm.client import create_llm_client
from medical_kg.logging import configure_logging
from medical_kg.models.source import SourceType
from medical_kg.pipeline.runner import PipelineRunner
from medical_kg.prompts import PromptRegistry, load_relation_vocabulary
from medical_kg.utils.statistics import collect_knowledge_statistics


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def _execute(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


def _settings(config: Path) -> AppSettings:
    settings = load_settings(config)
    configure_logging(settings.logging.level, settings.logging.json_output)
    return settings


def _repository(settings: AppSettings) -> KnowledgeRepository:
    settings.database.path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(
        settings.database.url,
        echo=settings.database.echo,
        connect_args={"timeout": settings.database.timeout},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def configure_sqlite(dbapi_connection: Any, _: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={int(settings.database.timeout * 1000)}")
        finally:
            cursor.close()

    return KnowledgeRepository(engine)


def _runner(settings: AppSettings, repository: KnowledgeRepository) -> PipelineRunner:
    extractor = BronzeExtractor(
        settings=settings,
        repository=repository,
        llm=create_llm_client(settings),
        prompts=PromptRegistry(settings.prompts.directory),
    )
    return PipelineRunner(
        repository=repository,
        loader=DocumentLoader(repository),
        extractor=extractor,
    )


def _print(value: Any) -> None:
    if hasattr(value, "__dict__"):
        value = value.__dict__
    print(json.dumps(value, indent=2, default=str))


async def _run_command(args: argparse.Namespace) -> Any:
    settings = _settings(args.config)
    repository = _repository(settings)
    runner: PipelineRunner | None = None
    try:
        if args.command == "init-db":
            await repository.create_schema()
            count = await repository.seed_relations(
                load_relation_vocabulary(settings.relations.vocabulary_file)
            )
            return {"relations_created": count}

        if args.command == "ingest":
            return await DocumentLoader(repository).ingest(
                args.source.resolve(), source_type=SourceType(args.source_type)
            )

        if args.command == "extract":
            runner = _runner(settings, repository)
            return await runner.extract(
                limit=args.limit,
                document_id=args.document_id,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )

        if args.command == "run":
            runner = _runner(settings, repository)
            ingested, extracted = await runner.run(
                args.source.resolve() if args.source else None,
                limit=args.limit,
                document_id=args.document_id,
                source_type=SourceType(args.source_type),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            return {
                "ingest": ingested.__dict__ if ingested else None,
                "extraction": extracted.__dict__,
            }

        if args.command == "retry-failed":
            count = await repository.retry_failed(
                stage_prefix="extract", document_id=args.document_id
            )
            return {"jobs_requeued": count}

        if args.command == "status":
            return await repository.job_status()

        if args.command == "stats":
            return await collect_knowledge_statistics(repository)

        raise ValueError(f"Unknown command: {args.command}")
    finally:
        try:
            if runner is not None:
                await runner.extractor.llm.aclose()
        finally:
            await repository.engine.dispose()


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))


def _add_source_type(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in SourceType],
        default=SourceType.RESEARCH.value,
    )


def _add_extraction_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--document-id")
    parser.add_argument("--chunk-size", type=_positive_int)
    parser.add_argument("--chunk-overlap", type=_nonnegative_int, default=0)
    _add_config(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m medical_kg",
        description="Medical literature knowledge graph pipeline",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init_db = commands.add_parser("init-db", help="Initialize the SQLite database schema")
    _add_config(init_db)

    ingest = commands.add_parser("ingest", help="Register local documents")
    ingest.add_argument("source", type=Path)
    _add_source_type(ingest)
    _add_config(ingest)

    extract = commands.add_parser("extract", help="Extract pending documents")
    _add_extraction_options(extract)

    run = commands.add_parser("run", help="Optionally ingest, then extract")
    run.add_argument("source", nargs="?", type=Path)
    _add_source_type(run)
    _add_extraction_options(run)

    retry = commands.add_parser("retry-failed", help="Requeue failed extraction jobs")
    retry.add_argument("--document-id")
    _add_config(retry)

    status = commands.add_parser("status", help="Show processing job counts")
    _add_config(status)

    stats = commands.add_parser("stats", help="Show read-only knowledge build statistics")
    _add_config(stats)

    commands.add_parser("canonicalize", help="Describe the gated Phase II operation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "canonicalize":
        print(
            "Canonicalization is scaffolded but intentionally gated until Bronze extraction is "
            "validated. Raw assertions are never modified; implement semantic resolvers before "
            "enabling."
        )
        return 0
    _print(_execute(_run_command(args)))
    return 0
