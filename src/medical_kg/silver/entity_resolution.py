from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalized_alias(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[\W_]+", " ", value).strip()


@dataclass(frozen=True)
class EntityCandidate:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...] = ()


class ConservativeEntityResolver:
    """Resolve exact normalized aliases only; ambiguity always remains unresolved."""

    def resolve(
        self, mention: str, entity_type: str, candidates: list[EntityCandidate]
    ) -> str | None:
        key = normalized_alias(mention)
        matches = {
            candidate.entity_id
            for candidate in candidates
            if candidate.entity_type == entity_type
            and key
            in {
                normalized_alias(candidate.canonical_name),
                *(normalized_alias(alias) for alias in candidate.aliases),
            }
        }
        return next(iter(matches)) if len(matches) == 1 else None
