from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field

from medical_kg.openalex.models import OpenAlexWork

MEDICAL_CORE_FIELDS = frozenset(
    {
        27,  # Medicine
        36,  # Health Professions
    }
)

BIOMEDICAL_FIELDS = frozenset(
    {
        13,  # Biochemistry, Genetics and Molecular Biology
        24,  # Immunology and Microbiology
        28,  # Neuroscience
        30,  # Pharmacology, Toxicology and Pharmaceutics
    }
)

MEDICAL_BROAD_FIELDS = MEDICAL_CORE_FIELDS | BIOMEDICAL_FIELDS


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


@dataclass(frozen=True)
class WorkFilter:
    allowed_field_ids: tuple[int, ...] | None = field(
        default_factory=lambda: tuple(sorted(MEDICAL_BROAD_FIELDS))
    )
    keywords: list[str] = field(default_factory=list)
    keyword_mode: str = "any"
    exclude_keywords: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    require_fulltext: bool = False

    def __post_init__(self) -> None:
        if self.keyword_mode not in {"any", "all"}:
            raise ValueError("keyword_mode must be 'any' or 'all'")

    def match(self, work: OpenAlexWork) -> tuple[bool, list[str]]:
        if self.allowed_field_ids is not None:
            work_field_ids = work.field_ids
            if not any(field_id in work_field_ids for field_id in self.allowed_field_ids):
                return False, []
        text = _normalize(f"{work.title}\n{work.abstract or ''}")
        normalized_keywords = [
            (keyword, _normalize(keyword)) for keyword in self.keywords if keyword.strip()
        ]
        matched = [original for original, keyword in normalized_keywords if keyword in text]
        if normalized_keywords:
            wanted = (
                len(matched) == len(normalized_keywords)
                if self.keyword_mode == "all"
                else bool(matched)
            )
            if not wanted:
                return False, []
        if any(
            _normalize(keyword) in text
            for keyword in self.exclude_keywords
            if keyword.strip()
        ):
            return False, []
        if self.sources:
            source_text = _normalize(json.dumps(work.sources, ensure_ascii=False))
            if not any(_normalize(selector) in source_text for selector in self.sources):
                return False, []
        if self.require_fulltext and not work.has_fulltext_hint:
            return False, []
        return True, matched
