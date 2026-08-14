from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from medical_kg.models.enums import StringEnum


class SourceType(StringEnum):
    TEXTBOOK = "textbook"
    GUIDELINES = "guidelines"
    RESEARCH = "research"


class KnowledgeSource(BaseModel):
    """A document contribution attached to a graph node or edge."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1, max_length=255)
    source_type: SourceType


def merge_sources(*groups: Iterable[KnowledgeSource | dict[str, Any]]) -> list[dict[str, str]]:
    """Validate, de-duplicate, and stably merge source lists."""

    merged: list[dict[str, str]] = []
    seen: set[tuple[str, SourceType]] = set()
    for group in groups:
        for raw_source in group:
            source = KnowledgeSource.model_validate(raw_source)
            key = (source.document_id, source.source_type)
            if key not in seen:
                seen.add(key)
                merged.append(source.model_dump(mode="json"))
    return merged
