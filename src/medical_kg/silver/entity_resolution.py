from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
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


class IndexedEntityResolver:
    """Resolve deterministic aliases without scanning every entity for every mention."""

    def __init__(self, candidates: list[EntityCandidate] | None = None) -> None:
        self._canonical_names: dict[str, str] = {}
        self._entity_types: dict[str, str] = {}
        self._aliases: defaultdict[str, set[str]] = defaultdict(set)
        self._external_ids: defaultdict[str, set[str]] = defaultdict(set)
        self._entities_by_type: defaultdict[str, set[str]] = defaultdict(set)
        self._exact: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        self._abbreviation_keys: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        self._abbreviation_form_keys: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        self._compact_forms: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        self._compact_abbreviations: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        for candidate in candidates or []:
            self.add_candidate(candidate)

    def add_candidate(self, candidate: EntityCandidate) -> None:
        entity_id = candidate.entity_id
        self._canonical_names.setdefault(entity_id, candidate.canonical_name)
        self._entity_types.setdefault(entity_id, candidate.entity_type)
        self._entities_by_type[candidate.entity_type].add(entity_id)
        self._external_ids[entity_id].update(candidate.external_ids)
        self._index_form(entity_id, candidate.entity_type, candidate.canonical_name)
        for alias in candidate.aliases:
            self.add_alias(entity_id, candidate.entity_type, alias)

    def add_alias(self, entity_id: str, entity_type: str, alias: str) -> None:
        if alias in self._aliases[entity_id]:
            return
        self._aliases[entity_id].add(alias)
        self._index_form(entity_id, entity_type, alias)

    def candidates(self, entity_type: str) -> list[EntityCandidate]:
        return [
            EntityCandidate(
                entity_id=entity_id,
                canonical_name=self._canonical_names[entity_id],
                entity_type=entity_type,
                aliases=tuple(sorted(self._aliases[entity_id])),
                external_ids=tuple(sorted(self._external_ids[entity_id])),
            )
            for entity_id in sorted(self._entities_by_type[entity_type])
        ]

    def resolve_decision(self, mention: str, entity_type: str) -> ResolutionDecision:
        key = normalized_alias(mention)
        exact_matches = self._exact[(entity_type, key)] if key else set()
        if len(exact_matches) == 1:
            return ResolutionDecision(next(iter(exact_matches)), "exact_alias", 1.0)
        if len(exact_matches) > 1:
            return ResolutionDecision(None, "ambiguous", 0.0)

        compact = _compact(mention)
        short_key = abbreviation_key(mention)
        if _looks_like_abbreviation(mention):
            abbreviation_matches = set(self._abbreviation_keys[(entity_type, compact)])
            abbreviation_matches.update(self._compact_forms[(entity_type, short_key)])
        else:
            abbreviation_matches = set(
                self._abbreviation_form_keys[(entity_type, compact)]
            )
            abbreviation_matches.update(
                self._compact_abbreviations[(entity_type, short_key)]
            )
        if len(abbreviation_matches) == 1:
            return ResolutionDecision(
                next(iter(abbreviation_matches)), "abbreviation", 0.98
            )
        if len(abbreviation_matches) > 1:
            return ResolutionDecision(None, "ambiguous", 0.0)

        synonym_matches: set[str] = set()
        for synonym_type, synonyms in DEFAULT_EQUIVALENT_ALIASES:
            if synonym_type != entity_type or key not in synonyms:
                continue
            for synonym in synonyms:
                synonym_matches.update(self._exact[(entity_type, synonym)])
        if len(synonym_matches) == 1:
            return ResolutionDecision(next(iter(synonym_matches)), "known_synonym", 0.99)
        return ResolutionDecision(None, "new", 1.0)

    def _index_form(self, entity_id: str, entity_type: str, form: str) -> None:
        normalized = normalized_alias(form)
        compact = _compact(form)
        short_key = abbreviation_key(form)
        if normalized:
            self._exact[(entity_type, normalized)].add(entity_id)
        if short_key:
            self._abbreviation_keys[(entity_type, short_key)].add(entity_id)
        if compact:
            self._compact_forms[(entity_type, compact)].add(entity_id)
        if _looks_like_abbreviation(form):
            if short_key:
                self._abbreviation_form_keys[(entity_type, short_key)].add(entity_id)
            if compact:
                self._compact_abbreviations[(entity_type, compact)].add(entity_id)
