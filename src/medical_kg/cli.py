from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Coroutine

import typer
from sqlalchemy.ext.asyncio import create_async_engine

from medical_kg.bronze.extraction import BronzeExtractor
from medical_kg.config import AppSettings, load_settings
from medical_kg.db.repository import KnowledgeRepository
from medical_kg.landing.loader import DocumentLoader
from medical_kg.llm.client import create_llm_client
from medical_kg.logging import configure_logging
from medical_kg.pipeline.runner import PipelineRunner
from medical_kg.prompts import PromptRegistry, load_relation_vocabulary


app = typer.Typer(no_args_is_help=True, help="Medical literature knowledge graph pipeline")


def _execute(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


def _settings(config: Path) -> AppSettings:
    settings = load_settings(config)
    configure_logging(settings.logging.level, settings.logging.json)
    return settings


def _repository(settings: AppSettings) -> KnowledgeRepository:
    engine = create_async_engine(
        settings.database.url, echo=settings.database.echo, pool_pre_ping=True
    )
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
    typer.echo(json.dumps(value, indent=2, default=str))


@app.command("init-db")
def init_db(config: Path = typer.Option(Path("config.yaml"), "--config")) -> None:
    async def operation() -> dict[str, int]:
        settings = _settings(config)
        repository = _repository(settings)
        try:
            await repository.create_schema()
            count = await repository.seed_relations(
                load_relation_vocabulary(settings.relations.vocabulary_file)
            )
            return {"relations_created": count}
        finally:
            await repository.engine.dispose()

    _print(_execute(operation()))


@app.command()
def ingest(
    source: Path,
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    async def operation() -> Any:
        settings = _settings(config)
        repository = _repository(settings)
        try:
            return await DocumentLoader(repository).ingest(source.resolve())
        finally:
            await repository.engine.dispose()

    _print(_execute(operation()))


@app.command()
def extract(
    limit: int | None = typer.Option(None, min=1),
    document_id: str | None = typer.Option(None, "--document-id"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    async def operation() -> Any:
        settings = _settings(config)
        repository = _repository(settings)
        try:
            return await _runner(settings, repository).extract(limit=limit, document_id=document_id)
        finally:
            await repository.engine.dispose()

    _print(_execute(operation()))


@app.command()
def run(
    source: Path | None = typer.Argument(None),
    limit: int | None = typer.Option(None, min=1),
    document_id: str | None = typer.Option(None, "--document-id"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    async def operation() -> Any:
        settings = _settings(config)
        repository = _repository(settings)
        try:
            ingested, extracted = await _runner(settings, repository).run(
                source.resolve() if source else None, limit=limit, document_id=document_id
            )
            return {
                "ingest": ingested.__dict__ if ingested else None,
                "extraction": extracted.__dict__,
            }
        finally:
            await repository.engine.dispose()

    _print(_execute(operation()))


@app.command("retry-failed")
def retry_failed(
    document_id: str | None = typer.Option(None, "--document-id"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    async def operation() -> dict[str, int]:
        settings = _settings(config)
        repository = _repository(settings)
        try:
            count = await repository.retry_failed(
                stage_prefix="extract", document_id=document_id
            )
            return {"jobs_requeued": count}
        finally:
            await repository.engine.dispose()

    _print(_execute(operation()))


@app.command()
def status(config: Path = typer.Option(Path("config.yaml"), "--config")) -> None:
    async def operation() -> Any:
        settings = _settings(config)
        repository = _repository(settings)
        try:
            return await repository.job_status()
        finally:
            await repository.engine.dispose()

    _print(_execute(operation()))


@app.command()
def canonicalize() -> None:
    """Describe the intentionally gated Phase II operation."""
    typer.echo(
        "Canonicalization is scaffolded but intentionally gated until Bronze extraction is "
        "validated. Raw assertions are never modified; implement semantic resolvers before "
        "enabling."
    )


if __name__ == "__main__":
    app()
