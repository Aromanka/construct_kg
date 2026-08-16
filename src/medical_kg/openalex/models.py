from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_WORK_ID = re.compile(r"^W\d+$", re.IGNORECASE)


def normalize_work_id(value: str) -> str:
    """Return the stable short OpenAlex Work ID (for example ``W2741809807``)."""

    candidate = value.rstrip("/").rsplit("/", 1)[-1].upper()
    if not _WORK_ID.fullmatch(candidate):
        raise ValueError(f"Invalid OpenAlex Work ID: {value!r}")
    return candidate


def restore_abstract(inverted_index: Any) -> str | None:
    """Restore OpenAlex's positional abstract representation to plain text."""

    if not isinstance(inverted_index, dict) or not inverted_index:
        return None
    positioned: dict[int, str] = {}
    for token, positions in inverted_index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                positioned.setdefault(position, token)
    if not positioned:
        return None
    return " ".join(positioned[position] for position in sorted(positioned)) or None


def _source_key(source: dict[str, Any]) -> str:
    return str(source.get("id") or source.get("display_name") or "")


def _collect_sources(raw: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    locations = [raw.get("primary_location"), *(raw.get("locations") or [])]
    for location in locations:
        source = location.get("source") if isinstance(location, dict) else None
        if not isinstance(source, dict):
            continue
        key = _source_key(source)
        if key and key not in seen:
            seen.add(key)
            sources.append(source)
    return sources


def _content_hints(raw: dict[str, Any]) -> tuple[bool, list[str]]:
    urls: list[str] = []
    has_content = raw.get("has_content")
    available = bool(raw.get("has_fulltext"))
    if isinstance(has_content, dict):
        available = available or any(bool(value) for value in has_content.values())
    elif isinstance(has_content, bool):
        available = available or has_content

    content_urls = raw.get("content_urls")
    if isinstance(content_urls, dict):
        for value in content_urls.values():
            if isinstance(value, str) and value:
                urls.append(value)
    locations = [
        raw.get("best_oa_location"),
        raw.get("primary_location"),
        *(raw.get("locations") or []),
    ]
    for location in locations:
        if not isinstance(location, dict):
            continue
        url = location.get("pdf_url")
        if isinstance(url, str) and url:
            urls.append(url)
    deduplicated = list(dict.fromkeys(urls))
    return available or bool(deduplicated), deduplicated


@dataclass(frozen=True)
class OpenAlexWork:
    work_id: str
    openalex_id: str
    title: str
    abstract: str | None
    doi: str | None
    publication_year: int | None
    language: str | None
    work_type: str | None
    sources: list[dict[str, Any]]
    primary_source_id: str | None
    primary_source_name: str | None
    has_fulltext_hint: bool
    fulltext_urls: list[str]
    raw: dict[str, Any] = field(repr=False)
    snapshot_file: str = ""
    snapshot_line: int = 0

    @property
    def document_id(self) -> str:
        return f"openalex:{self.work_id}"

    @classmethod
    def from_raw(
        cls, raw: dict[str, Any], *, snapshot_file: str = "", snapshot_line: int = 0
    ) -> OpenAlexWork:
        openalex_id = str(raw.get("id") or "")
        work_id = normalize_work_id(openalex_id)
        primary = raw.get("primary_location")
        primary_source = primary.get("source") if isinstance(primary, dict) else None
        if not isinstance(primary_source, dict):
            primary_source = {}
        available, urls = _content_hints(raw)
        year = raw.get("publication_year")
        return cls(
            work_id=work_id,
            openalex_id=openalex_id or f"https://openalex.org/{work_id}",
            title=str(raw.get("title") or raw.get("display_name") or "").strip(),
            abstract=restore_abstract(raw.get("abstract_inverted_index")),
            doi=str(raw["doi"]) if raw.get("doi") else None,
            publication_year=year if isinstance(year, int) else None,
            language=str(raw["language"]) if raw.get("language") else None,
            work_type=str(raw["type"]) if raw.get("type") else None,
            sources=_collect_sources(raw),
            primary_source_id=(
                str(primary_source["id"]) if primary_source.get("id") else None
            ),
            primary_source_name=(
                str(primary_source["display_name"])
                if primary_source.get("display_name")
                else None
            ),
            has_fulltext_hint=available,
            fulltext_urls=urls,
            raw=raw,
            snapshot_file=snapshot_file,
            snapshot_line=snapshot_line,
        )
