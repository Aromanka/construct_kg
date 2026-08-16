from __future__ import annotations

import gzip
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from medical_kg.openalex.models import OpenAlexWork

logger = logging.getLogger(__name__)
_PART_FILE = re.compile(r"^part_\d+\.gz$")


class OpenAlexSnapshot:
    """Stream OpenAlex snapshot entities without materializing gzip files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.data_dir = self._find_data_dir()

    def _find_data_dir(self) -> Path:
        candidates = (self.root / "data", self.root, self.root / "data" / "jsonl")
        for candidate in candidates:
            if (candidate / "works").is_dir():
                return candidate
        raise FileNotFoundError(
            f"Cannot find data/works below OpenAlex snapshot root {self.root}"
        )

    def entity_files(self, entity: str) -> list[Path]:
        entity_dir = self.data_dir / entity
        if not entity_dir.is_dir():
            raise FileNotFoundError(f"OpenAlex entity directory does not exist: {entity_dir}")
        # A strict filename intentionally ignores interrupted/corrupt copies such as
        # part_0001.gz.A1b2C3, which occur in the supplied directory listing.
        files = [
            path
            for path in entity_dir.rglob("*.gz")
            if path.is_file() and _PART_FILE.fullmatch(path.name)
        ]
        files = sorted(files, key=lambda path: path.relative_to(entity_dir).as_posix())
        manifest = entity_dir / "manifest"
        if not manifest.is_file():
            return files
        manifest_references = self._manifest_references(manifest)
        if not manifest_references:
            logger.warning("could not parse OpenAlex manifest; using strict file discovery")
            return files
        ordered: list[Path] = []
        for reference in manifest_references:
            normalized = unquote(urlparse(reference).path).replace("\\", "/")
            match = next(
                (
                    path
                    for path in files
                    if normalized.endswith(path.relative_to(entity_dir).as_posix())
                ),
                None,
            )
            if match is not None and match not in ordered:
                ordered.append(match)
        # A successfully parsed manifest is authoritative. Fall back only when its schema
        # yielded no local matches (for example, a future format change).
        return ordered or files

    @staticmethod
    def _manifest_references(path: Path) -> list[str]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return []
        references: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)
            elif isinstance(value, str) and _PART_FILE.fullmatch(
                Path(urlparse(value).path).name
            ):
                references.append(value)

        visit(payload)
        return references

    def iter_raw(self, entity: str) -> Iterator[tuple[dict[str, Any], str, int]]:
        for path in self.entity_files(entity):
            relative = path.relative_to(self.root).as_posix()
            try:
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        if not isinstance(payload, dict):
                            raise ValueError("JSONL record is not an object")
                        yield payload, relative, line_number
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                logger.exception("failed to read OpenAlex part", extra={"path": str(path)})
                raise

    def iter_works(self, *, max_works: int | None = None) -> Iterator[OpenAlexWork]:
        count = 0
        for raw, relative, line_number in self.iter_raw("works"):
            try:
                work = OpenAlexWork.from_raw(
                    raw, snapshot_file=relative, snapshot_line=line_number
                )
            except ValueError:
                logger.warning(
                    "skipping Work with invalid ID",
                    extra={"snapshot_file": relative, "snapshot_line": line_number},
                )
                continue
            yield work
            count += 1
            if max_works is not None and count >= max_works:
                return
