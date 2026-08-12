from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    document_id: str = Field(min_length=1, max_length=255)
    file_path: Path
    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str | None = None
    doi: str | None = None
    pmid: str | None = None

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
    ) -> "DocumentInput":
        return cls(
            document_id=document_id,
            file_path=file_path,
            title=title,
            doi=doi,
            pmid=pmid,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

