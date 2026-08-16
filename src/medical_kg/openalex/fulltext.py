from __future__ import annotations

import gzip
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from medical_kg.openalex.models import OpenAlexWork

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedFullText:
    text: str | None
    path: Path | None
    status: str


def _xml_text(payload: bytes) -> str:
    root = ET.fromstring(payload)
    return " ".join(part.strip() for part in root.itertext() if part.strip())


def read_fulltext(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".xml", ".gz"]:
        return _xml_text(gzip.decompress(path.read_bytes()))
    if path.suffix.lower() in {".xml", ".grobid-xml"}:
        return _xml_text(path.read_bytes())
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return str(
                payload.get("full_text") or payload.get("content") or payload.get("text") or ""
            )
        return ""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF full text requires `pip install medical-kg[pdf]`") from error
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raise ValueError(f"Unsupported full-text file: {path}")


class FullTextResolver:
    """Resolve selected full text locally, optionally using OpenAlex's content service."""

    suffixes = (".grobid-xml", ".xml", ".xml.gz", ".txt", ".md", ".json", ".pdf")

    def __init__(
        self,
        *,
        output_dir: Path,
        local_dir: Path | None = None,
        download: bool = False,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_dir = local_dir.resolve() if local_dir else None
        self.download = download
        self.api_key = api_key
        self.client = (
            httpx.AsyncClient(timeout=timeout, follow_redirects=True) if download else None
        )
        self._remote_disabled_status: str | None = None

    def _local_candidates(self, work_id: str) -> list[Path]:
        if self.local_dir is None:
            return []
        candidates: list[Path] = []
        for suffix in self.suffixes:
            candidates.extend(
                (
                    self.local_dir / f"{work_id}{suffix}",
                    self.local_dir / work_id / f"{work_id}{suffix}",
                )
            )
        return candidates

    async def resolve(self, work: OpenAlexWork) -> ResolvedFullText:
        for path in self._local_candidates(work.work_id):
            if path.is_file():
                text = read_fulltext(path)
                return ResolvedFullText(text.strip() or None, path, "local")
        if not self.download or self.client is None:
            return ResolvedFullText(None, None, "not_found")
        if self._remote_disabled_status is not None:
            return ResolvedFullText(None, None, self._remote_disabled_status)

        urls = [
            f"https://content.openalex.org/works/{work.work_id}.grobid-xml",
            f"https://content.openalex.org/works/{work.work_id}.pdf",
        ]
        # API-only content_urls may point to the same trusted OpenAlex content host.
        urls.extend(
            url for url in work.fulltext_urls if urlparse(url).hostname == "content.openalex.org"
        )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        parameters = {"api_key": self.api_key} if self.api_key else {}
        download_failed = False
        for url in dict.fromkeys(urls):
            try:
                response = await self.client.get(url, headers=headers, params=parameters)
            except httpx.HTTPError as error:
                download_failed = True
                logger.warning("OpenAlex full-text download failed for %s: %s", work.work_id, error)
                continue
            if response.status_code == 404:
                continue
            if response.status_code in {401, 402, 403, 429}:
                self._remote_disabled_status = (
                    "quota_unavailable" if response.status_code in {402, 429} else "unauthorized"
                )
                logger.warning(
                    "OpenAlex full-text service unavailable (%s); skipping remote downloads",
                    response.status_code,
                )
                return ResolvedFullText(None, None, self._remote_disabled_status)
            if response.is_error:
                download_failed = True
                logger.warning(
                    "OpenAlex full-text download returned HTTP %s for %s",
                    response.status_code,
                    work.work_id,
                )
                continue
            is_pdf = urlparse(url).path.lower().endswith(".pdf") or response.content[:4] == b"%PDF"
            suffix = ".pdf" if is_pdf else ".grobid-xml"
            path = self.output_dir / f"{work.work_id}{suffix}"
            try:
                path.write_bytes(response.content)
                text = read_fulltext(path)
            except (OSError, RuntimeError, ValueError, ET.ParseError) as error:
                download_failed = True
                logger.warning("OpenAlex full-text parsing failed for %s: %s", work.work_id, error)
                continue
            return ResolvedFullText(text.strip() or None, path, "downloaded")
        return ResolvedFullText(None, None, "download_failed" if download_failed else "not_found")

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()
