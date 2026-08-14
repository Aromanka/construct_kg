from __future__ import annotations

import hashlib
import json
from typing import Any


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
        "qualifiers": qualifiers,
        "negated": negated,
        "speculative": speculative,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
