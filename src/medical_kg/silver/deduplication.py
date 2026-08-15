from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

EVIDENCE_LEVEL_QUALIFIERS = frozenset(
    {
        "effect_size",
        "statistical_significance",
        "study_type",
    }
)


def canonical_qualifiers(qualifiers: Mapping[str, Any]) -> dict[str, Any]:
    """Keep fact-defining qualifiers and remove formatting/evidence-only variation."""

    def normalize(value: Any) -> Any:
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip().casefold()
        if isinstance(value, dict):
            return {
                key: normalized
                for key, item in sorted(value.items())
                if (normalized := normalize(item)) not in (None, "", [], {})
            }
        if isinstance(value, list):
            normalized = [normalize(item) for item in value]
            return sorted(
                (item for item in normalized if item not in (None, "", [], {})),
                key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False),
            )
        return value

    return normalize(
        {
            key: value
            for key, value in qualifiers.items()
            if key not in EVIDENCE_LEVEL_QUALIFIERS
        }
    )


def normalized_assertion_identity(
    *,
    subject_entity_id: str,
    canonical_relation_id: str,
    object_entity_id: str,
    qualifiers: dict[str, Any],
    negated: bool,
    speculative: bool,
) -> str:
    value = {
        "subject": subject_entity_id,
        "relation": canonical_relation_id,
        "object": object_entity_id,
        "qualifiers": canonical_qualifiers(qualifiers),
        "negated": negated,
        "speculative": speculative,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
