from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RelationMapping:
    canonical_relation: str
    confidence: float


class ExactRelationNormalizer:
    """Conservative rule baseline with an explicit OTHER fallback."""

    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = {item.casefold(): item for item in vocabulary}

    def normalize(self, detailed_relation: str) -> RelationMapping:
        candidate = re.sub(r"[\W_]+", "_", detailed_relation.casefold()).strip("_")
        if candidate in self.vocabulary:
            return RelationMapping(self.vocabulary[candidate], 1.0)

        text = re.sub(r"[\W_]+", " ", detailed_relation.casefold()).strip()
        # Ordered from specific to broad so risk and direction are not erased by "associated".
        rules = (
            (r"\b(decreas|reduc|lower).*\brisk\b", "decreases_risk_of", 0.94),
            (r"\b(increas|rais|elevat|higher).*\brisk\b", "increases_risk_of", 0.94),
            (
                r"\b(negative|invers).*\b(associat|correlat|link)",
                "negatively_associated_with",
                0.92,
            ),
            (
                r"\b(positive|direct).*\b(associat|correlat|link)",
                "positively_associated_with",
                0.92,
            ),
            (r"\b(upregulat|increase[sd]? expression|promot.*expression)", "upregulates", 0.94),
            (
                r"\b(downregulat|decrease[sd]? expression|suppress.*expression)",
                "downregulates",
                0.94,
            ),
            (r"\b(associat|linked? to|relationship with)\b", "associated_with", 0.88),
            (r"\bcorrelat", "correlated_with", 0.9),
            (r"\b(caus|result(?:ed|s)? in|lead(?:s|ing)? to)\b", "causes", 0.9),
            (r"\bcontribut", "contributes_to", 0.9),
            (r"\b(treat|therapeutic for)", "treats", 0.92),
            (r"\b(ameliorat|alleviat|improv)", "ameliorates", 0.88),
            (r"\b(worsen|exacerbat)", "worsens", 0.92),
            (r"\b(prevent|avoid)", "prevents", 0.9),
            (r"\bprotect", "protects_against", 0.9),
            (r"\bactivat", "activates", 0.92),
            (r"\binhibit", "inhibits", 0.92),
            (r"\bupregulat", "upregulates", 0.94),
            (r"\bdownregulat", "downregulates", 0.94),
            (r"\bregulat", "regulates", 0.86),
            (r"\binteract", "interacts_with", 0.9),
            (r"\bbind", "binds_to", 0.92),
            (r"\bexpress", "expressed_in", 0.82),
            (r"\b(locat|localiz)", "located_in", 0.88),
            (r"\bpart of\b", "part_of", 0.94),
            (r"\b(include|contain|compris)", "includes", 0.88),
            (r"\bbiomarker", "biomarker_of", 0.86),
            (r"\bpredict", "predicts", 0.9),
            (r"\bdiagnos", "diagnoses", 0.9),
            (r"\bproduc", "produces", 0.88),
            (r"\bmetaboli[sz]", "metabolizes", 0.9),
            (r"\bconvert.*\bto\b", "converts_to", 0.9),
            (r"\brequir", "required_for", 0.88),
            (r"\bpromot", "promotes", 0.86),
            (r"\bsuppress", "suppresses", 0.86),
            (r"\b(increas|higher|elevat)", "increases", 0.82),
            (r"\b(decreas|lower|reduc)", "decreases", 0.82),
        )
        for pattern, canonical, confidence in rules:
            key = canonical.casefold()
            if key in self.vocabulary and re.search(pattern, text):
                return RelationMapping(self.vocabulary[key], confidence)
        other = self.vocabulary.get("other", "OTHER")
        return RelationMapping(other, 0.0)
