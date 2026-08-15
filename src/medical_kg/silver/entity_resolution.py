from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


def normalized_alias(value: str) -> str:
    """Normalize surface variation while preserving the words that carry identity."""

    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[\W_]+", " ", value).strip()


def abbreviation_key(value: str) -> str:
    """Return an acronym-like key (type 2 diabetes mellitus becomes t2dm)."""

    ignored = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
    tokens = normalized_alias(value).split()
    return "".join(
        token if token.isdigit() else token[0]
        for token in tokens
        if token not in ignored
    )


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", value).casefold())


def _looks_like_abbreviation(value: str) -> bool:
    compact = _compact(value)
    words = normalized_alias(value).split()
    return bool(compact) and len(compact) <= 12 and (
        len(words) == 1 or value.isupper() or any(character.isdigit() for character in value)
    )


@dataclass(frozen=True)
class EntityCandidate:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    external_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredEntityCandidate:
    candidate: EntityCandidate
    score: float
    reason: str


@dataclass(frozen=True)
class ResolutionDecision:
    entity_id: str | None
    method: str
    confidence: float


# High-precision, type-scoped biomedical synonym sets. Broad semantic similarity remains
# candidate retrieval only and never becomes an automatic merge.
DEFAULT_EQUIVALENT_ALIASES: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "DISEASE",
        frozenset(
            {
                "t2d",
                "t2dm",
                "type 2 diabetes",
                "type 2 diabetes mellitus",
                "type ii diabetes",
                "type ii diabetes mellitus",
            }
        ),
    ),
    (
        "DISEASE",
        frozenset({"dkd", "diabetic kidney disease", "diabetic nephropathy"}),
    ),
)

PREFERRED_CANONICAL_NAMES: dict[tuple[str, str], str] = {
    (entity_type, alias): preferred
    for entity_type, aliases, preferred in (
        (
            "DISEASE",
            DEFAULT_EQUIVALENT_ALIASES[0][1],
            "type 2 diabetes mellitus",
        ),
        (
            "DISEASE",
            DEFAULT_EQUIVALENT_ALIASES[1][1],
            "diabetic kidney disease",
        ),
    )
    for alias in aliases
}


def preferred_canonical_name(mention: str, entity_type: str) -> str:
    """Choose a stable name only for explicitly curated synonym groups."""

    stripped = mention.strip()
    return PREFERRED_CANONICAL_NAMES.get(
        (entity_type, normalized_alias(stripped)), stripped
    )


class CandidateRetriever:
    """Retrieve type-compatible candidates without deciding semantic identity."""

    def __init__(self, *, top_k: int = 8, minimum_score: float = 0.35) -> None:
        self.top_k = top_k
        self.minimum_score = minimum_score

    def retrieve(
        self, mention: str, entity_type: str, candidates: list[EntityCandidate]
    ) -> list[ScoredEntityCandidate]:
        mention_key = normalized_alias(mention)
        mention_tokens = set(mention_key.split())
        scored: list[ScoredEntityCandidate] = []
        for candidate in candidates:
            if candidate.entity_type != entity_type:
                continue
            forms = (candidate.canonical_name, *candidate.aliases)
            best_score = 0.0
            best_reason = "lexical"
            for form in forms:
                form_key = normalized_alias(form)
                if mention_key and mention_key == form_key:
                    best_score, best_reason = 1.0, "exact_alias"
                    break
                if _looks_like_abbreviation(mention) or _looks_like_abbreviation(form):
                    if (
                        _compact(mention) == abbreviation_key(form)
                        or _compact(form) == abbreviation_key(mention)
                    ):
                        best_score, best_reason = 0.98, "abbreviation"
                        break
                form_tokens = set(form_key.split())
                union = mention_tokens | form_tokens
                jaccard = len(mention_tokens & form_tokens) / len(union) if union else 0.0
                sequence = SequenceMatcher(None, mention_key, form_key).ratio()
                score = 0.6 * jaccard + 0.4 * sequence
                if score > best_score:
                    best_score = score
            if best_score >= self.minimum_score:
                scored.append(ScoredEntityCandidate(candidate, best_score, best_reason))
        scored.sort(key=lambda item: (-item.score, item.candidate.entity_id))
        return scored[: self.top_k]


class ConservativeEntityResolver:
    """High-precision resolver; lexical closeness alone never causes a merge."""

    def resolve_decision(
        self, mention: str, entity_type: str, candidates: list[EntityCandidate]
    ) -> ResolutionDecision:
        key = normalized_alias(mention)
        exact_matches = {
            candidate.entity_id
            for candidate in candidates
            if candidate.entity_type == entity_type
            and key
            in {
                normalized_alias(candidate.canonical_name),
                *(normalized_alias(alias) for alias in candidate.aliases),
            }
        }
        if len(exact_matches) == 1:
            return ResolutionDecision(next(iter(exact_matches)), "exact_alias", 1.0)
        if len(exact_matches) > 1:
            return ResolutionDecision(None, "ambiguous", 0.0)

        abbreviation_matches: set[str] = set()
        for candidate in candidates:
            if candidate.entity_type != entity_type:
                continue
            for form in (candidate.canonical_name, *candidate.aliases):
                if (_looks_like_abbreviation(mention) or _looks_like_abbreviation(form)) and (
                    _compact(mention) == abbreviation_key(form)
                    or _compact(form) == abbreviation_key(mention)
                ):
                    abbreviation_matches.add(candidate.entity_id)
        if len(abbreviation_matches) == 1:
            return ResolutionDecision(
                next(iter(abbreviation_matches)), "abbreviation", 0.98
            )
        if len(abbreviation_matches) > 1:
            return ResolutionDecision(None, "ambiguous", 0.0)

        synonym_matches: set[str] = set()
        for synonym_type, aliases in DEFAULT_EQUIVALENT_ALIASES:
            if synonym_type != entity_type or key not in aliases:
                continue
            for candidate in candidates:
                if candidate.entity_type != entity_type:
                    continue
                forms = {
                    normalized_alias(candidate.canonical_name),
                    *(normalized_alias(alias) for alias in candidate.aliases),
                }
                if forms & aliases:
                    synonym_matches.add(candidate.entity_id)
        if len(synonym_matches) == 1:
            return ResolutionDecision(next(iter(synonym_matches)), "known_synonym", 0.99)
        return ResolutionDecision(None, "new", 1.0)

    def resolve(
        self, mention: str, entity_type: str, candidates: list[EntityCandidate]
    ) -> str | None:
        return self.resolve_decision(mention, entity_type, candidates).entity_id
