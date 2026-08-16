from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from medical_kg.cli import _repository, _runner, _settings
from medical_kg.landing.loader import DocumentLoader
from medical_kg.models.source import SourceType
from medical_kg.openalex.catalog import OpenAlexCatalog
from medical_kg.openalex.filtering import WorkFilter
from medical_kg.openalex.fulltext import FullTextResolver
from medical_kg.openalex.pipeline import OpenAlexPipeline, SelectionOptions
from medical_kg.openalex.screening import CompatibleWorkScreener
from medical_kg.openalex.snapshot import OpenAlexSnapshot
from medical_kg.prompts import load_relation_vocabulary


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("data/openalex"),
        help="Feature output directory",
    )


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("snapshot", type=Path, help="OpenAlex snapshot root")
    _add_workspace(parser)
    parser.add_argument(
        "--keyword", action="append", default=[], help="Repeatable title/abstract keyword"
    )
    parser.add_argument("--keyword-mode", choices=("any", "all"), default="any")
    parser.add_argument("--exclude-keyword", action="append", default=[])
    parser.add_argument(
        "--source", action="append", default=[], help="Repeatable source ID/name/type selector"
    )
    parser.add_argument(
        "--require-fulltext",
        action="store_true",
        help="Require a snapshot full-text availability hint",
    )
    parser.add_argument(
        "--llm-prompt", help="Relevance instruction; use @path to read a UTF-8 file"
    )
    parser.add_argument("--llm-batch-size", type=_positive, default=20)
    parser.add_argument(
        "--include-id", action="append", default=[], help="Always select this Work ID"
    )
    parser.add_argument(
        "--all",
        dest="select_all",
        action="store_true",
        help="Explicitly allow selection without any filter",
    )
    parser.add_argument("--max-works", type=_positive, help="Testing/debug scan cap")
    parser.add_argument("--max-candidates", type=_nonnegative)
    parser.add_argument("--max-selected", type=_positive)
    parser.add_argument(
        "--enrich-sources", action="store_true", help="Join full records from data/sources"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))


def _add_materialization(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fulltext-dir", type=Path, help="Local W{id}.xml/.txt/.json/.pdf directory"
    )
    parser.add_argument(
        "--download-fulltext",
        action="store_true",
        help="Fetch selected content from content.openalex.org",
    )
    parser.add_argument("--openalex-api-key", default=os.getenv("OPENALEX_API_KEY"))
    parser.add_argument(
        "--content-mode",
        choices=("fulltext", "abstract", "fulltext-or-abstract"),
        default="fulltext-or-abstract",
    )
    parser.add_argument("--materialize-limit", type=_positive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python openalex_pipeline.py",
        description="Stream, screen, materialize, and extract OpenAlex literature",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select", help="Screen snapshot Works into a durable catalog")
    _add_selection(select)

    run = commands.add_parser(
        "run", help="Select and prepare documents; optionally extract relations"
    )
    _add_selection(run)
    _add_materialization(run)
    run.add_argument(
        "--extract", action="store_true", help="Ingest documents and run KG extraction"
    )
    run.add_argument("--chunk-size", type=_positive)
    run.add_argument("--chunk-overlap", type=_nonnegative, default=0)

    add = commands.add_parser("add", help="Add explicit stable Work IDs from the snapshot")
    add.add_argument("snapshot", type=Path)
    add.add_argument("work_ids", nargs="+")
    _add_workspace(add)
    add.add_argument("--max-works", type=_positive)

    materialize = commands.add_parser(
        "materialize", help="Prepare selected catalog Works as KG documents"
    )
    materialize.add_argument("snapshot", type=Path)
    _add_workspace(materialize)
    _add_materialization(materialize)

    show = commands.add_parser("show", help="Look up retained metadata by stable Work ID")
    show.add_argument("work_id")
    _add_workspace(show)
    return parser


def _instruction(value: str | None) -> str | None:
    if value and value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8").strip()
    return value.strip() if value else None


def _catalog(workspace: Path) -> OpenAlexCatalog:
    return OpenAlexCatalog(workspace.resolve() / "catalog.sqlite3")


def _selection_options(args: argparse.Namespace) -> SelectionOptions:
    instruction = _instruction(args.llm_prompt)
    has_filter = any(
        (
            args.keyword,
            args.source,
            args.require_fulltext,
            instruction,
            args.include_id,
        )
    )
    if not has_filter and not args.select_all:
        raise ValueError(
            "At least one filter is required; pass --all to intentionally scan/select all Works"
        )
    only_explicit = bool(args.include_id) and not any(
        (args.keyword, args.source, args.require_fulltext, instruction, args.select_all)
    )
    return SelectionOptions(
        work_filter=WorkFilter(
            keywords=args.keyword,
            keyword_mode=args.keyword_mode,
            exclude_keywords=args.exclude_keyword,
            sources=args.source,
            require_fulltext=args.require_fulltext,
        ),
        llm_instruction=instruction,
        llm_batch_size=args.llm_batch_size,
        include_ids=set(args.include_id),
        max_works=args.max_works,
        max_candidates=0 if only_explicit and args.max_candidates is None else args.max_candidates,
        max_selected=args.max_selected,
    )


async def _extract(args: argparse.Namespace, documents: list[Path]) -> dict[str, Any]:
    if not documents:
        return {
            "ingest": {
                "discovered": 0,
                "created": 0,
                "changed": 0,
                "unchanged": 0,
                "failed": 0,
            },
            "extraction": None,
        }
    settings = _settings(args.config)
    repository = _repository(settings)
    runner = _runner(settings, repository)
    try:
        await repository.create_schema()
        await repository.seed_relations(
            load_relation_vocabulary(settings.relations.vocabulary_file)
        )
        loader = DocumentLoader(repository)
        ingest_totals = {
            "discovered": 0,
            "created": 0,
            "changed": 0,
            "unchanged": 0,
            "failed": 0,
        }
        document_ids = []
        for path in documents:
            ingest = await loader.ingest(path, source_type=SourceType.RESEARCH)
            for key in ingest_totals:
                ingest_totals[key] += getattr(ingest, key)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("document_id"):
                document_ids.append(str(payload["document_id"]))
        extraction = None
        if document_ids:
            extraction = await runner.extract(
                document_ids=document_ids,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        return {
            "ingest": ingest_totals,
            "extraction": asdict(extraction) if extraction is not None else None,
        }
    finally:
        await runner.extractor.llm.aclose()
        await repository.engine.dispose()


async def _materialize(
    args: argparse.Namespace, pipeline: OpenAlexPipeline
) -> dict[str, Any]:
    resolver = FullTextResolver(
        output_dir=args.workspace.resolve() / "fulltext",
        local_dir=args.fulltext_dir,
        download=args.download_fulltext,
        api_key=args.openalex_api_key,
    )
    try:
        result = await pipeline.materialize(
            resolver=resolver,
            content_mode=args.content_mode,
            limit=args.materialize_limit,
        )
        return asdict(result)
    finally:
        await resolver.aclose()


async def _run(args: argparse.Namespace) -> Any:
    if args.command == "show":
        with _catalog(args.workspace) as catalog:
            row = catalog.get(args.work_id)
            if row is None:
                raise LookupError(f"Work is not present in the catalog: {args.work_id}")
            result = dict(row)
            json_fields = (
                "sources_json",
                "fulltext_urls_json",
                "raw_json",
                "matched_keywords_json",
            )
            for key in json_fields:
                result[key.removesuffix("_json")] = json.loads(result.pop(key))
            return result

    snapshot = OpenAlexSnapshot(args.snapshot)
    with _catalog(args.workspace) as catalog:
        pipeline = OpenAlexPipeline(
            snapshot=snapshot, catalog=catalog, workspace=args.workspace
        )
        if args.command == "add":
            result = await pipeline.add(set(args.work_ids), max_works=args.max_works)
            return asdict(result)
        if args.command == "materialize":
            return await _materialize(args, pipeline)

        options = _selection_options(args)
        screener = None
        try:
            if options.llm_instruction:
                screener = CompatibleWorkScreener(_settings(args.config))
            selection = await pipeline.select(options, screener=screener)
        finally:
            if screener is not None:
                await screener.aclose()
        output: dict[str, Any] = {"selection": asdict(selection)}
        if args.enrich_sources:
            output["sources_enriched"] = pipeline.enrich_sources()
        if args.command == "run":
            output["materialization"] = await _materialize(args, pipeline)
            if args.extract:
                output["knowledge_extraction"] = await _extract(
                    args,
                    catalog.selected_materialized_paths(limit=args.materialize_limit),
                )
        return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(_run(args))
    if is_dataclass(result):
        result = asdict(result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0
