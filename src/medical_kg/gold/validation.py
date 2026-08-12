from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str


def validate_assertion_candidate(
    candidate: dict[str, Any], *, source_content: str, known_relations: set[str]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field in ("subject_entity_id", "object_entity_id"):
        if not candidate.get(field):
            issues.append(ValidationIssue(field, "canonical entity is required"))
    if not str(candidate.get("detailed_relation", "")).strip():
        issues.append(ValidationIssue("detailed_relation", "must be non-empty"))
    confidence = candidate.get("llm_confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        issues.append(ValidationIssue("llm_confidence", "must be between 0 and 1"))
    evidence = candidate.get("evidence_text")
    if not isinstance(evidence, str) or evidence not in source_content:
        issues.append(ValidationIssue("evidence_text", "must occur exactly in the source"))
    relation = candidate.get("canonical_relation")
    if relation not in known_relations:
        issues.append(ValidationIssue("canonical_relation", "must exist in the vocabulary"))
    return issues

