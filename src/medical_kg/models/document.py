from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from medical_kg.models.source import SourceType


class DocumentInput(BaseModel):
    document_id: str = Field(min_length=1, max_length=255)
    file_path: Path
    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str | None = None
    doi: str | None = None
    pmid: str | None = None
    source_type: SourceType = SourceType.RESEARCH

    @classmethod
    def from_content(
        cls,
        *,
        document_id: str,
        file_path: Path,
        content: str,
        title: str | None = None,
        doi: str | None = None,
        pmid: str | None = None,
        source_type: SourceType = SourceType.RESEARCH,
    ) -> DocumentInput:
        return cls(
            document_id=document_id,
            file_path=file_path,
            title=title,
            doi=doi,
            pmid=pmid,
            source_type=source_type,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
