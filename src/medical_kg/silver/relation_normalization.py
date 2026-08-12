from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationMapping:
    canonical_relation: str
    confidence: float


class ExactRelationNormalizer:
    """Safe deterministic baseline; semantic mappings are delegated to an LLM in Phase II."""

    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = set(vocabulary)

    def normalize(self, detailed_relation: str) -> RelationMapping:
        candidate = detailed_relation.strip().lower().replace(" ", "_")
        if candidate in self.vocabulary:
            return RelationMapping(candidate, 1.0)
        return RelationMapping("OTHER", 0.0)

