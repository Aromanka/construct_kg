from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg.db.repository import KnowledgeRepository
from medical_kg.models.document import DocumentInput
from medical_kg.models.source import SourceType

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    discovered: int = 0
    created: int = 0
    changed: int = 0
    unchanged: int = 0
    failed: int = 0


class DocumentLoader:
    """Load UTF-8 text, Markdown, JSON, and optionally PDF documents."""

    supported_suffixes = {".txt", ".md", ".json", ".pdf"}

    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository

    async def ingest(
        self,
        source: Path,
        *,
        source_type: SourceType = SourceType.RESEARCH,
        progress: Callable[[IngestResult], None] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        for path in self._discover(source):
            result.discovered += 1
            try:
                document = self.load_file(
                    path,
                    source if source.is_dir() else source.parent,
                    default_source_type=source_type,
                )
                created, changed = await self.repository.register_document(document)
                if created:
                    result.created += 1
                elif changed:
                    result.changed += 1
                else:
                    result.unchanged += 1
            except Exception:
                result.failed += 1
                logger.exception(
                    "document ingestion failed",
                    extra={"document_id": str(path), "stage": "landing"},
                )
            if progress is not None:
                progress(result)
        return result

    def count(self, source: Path) -> int:
        """Count documents that would be discovered by ``ingest``."""

        return sum(1 for _ in self._discover(source))

    def _discover(self, source: Path) -> Iterable[Path]:
        if source.is_file():
            if source.suffix.lower() not in self.supported_suffixes:
                raise ValueError(f"Unsupported document type: {source.suffix}")
            yield source.resolve()
            return
        if not source.is_dir():
            raise FileNotFoundError(source)
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.supported_suffixes:
                yield path.resolve()

    def load_file(
        self,
        path: Path,
        root: Path,
        *,
        default_source_type: SourceType = SourceType.RESEARCH,
    ) -> DocumentInput:
        suffix = path.suffix.lower()
        metadata: dict[str, Any] = {}
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, str):
                content = payload
            elif isinstance(payload, dict):
                content = str(payload.get("content") or payload.get("text") or "")
                metadata = payload
            else:
                raise ValueError(f"Expected JSON string or object in {path}")
        elif suffix == ".pdf":
            content = self._read_pdf(path)
        else:
            content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Document is empty: {path}")
        relative = path.relative_to(root.resolve()).as_posix()
        stable_id = str(metadata.get("document_id") or relative)
        return DocumentInput.from_content(
            document_id=stable_id,
            file_path=path,
            content=content,
            title=metadata.get("title") or path.stem,
            doi=metadata.get("doi"),
            pmid=str(metadata["pmid"]) if metadata.get("pmid") is not None else None,
            source_type=metadata.get("source_type", default_source_type),
        )

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF support requires `pip install medical-kg[pdf]`") from error
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
