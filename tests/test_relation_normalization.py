from medical_kg.silver.relation_normalization import ExactRelationNormalizer


def test_relation_normalizer_maps_surface_phrases_by_specificity() -> None:
    normalizer = ExactRelationNormalizer(
        [
            "associated_with",
            "increases_risk_of",
            "negatively_associated_with",
            "upregulates",
            "OTHER",
        ]
    )

    assert (
        normalizer.normalize(
            "was independently associated with an increased risk of"
        ).canonical_relation
        == "increases_risk_of"
    )
    assert (
        normalizer.normalize(
            "was negatively associated with"
        ).canonical_relation
        == "negatively_associated_with"
    )
    assert (
        normalizer.normalize("promoted the expression of").canonical_relation
        == "upregulates"
    )


def test_relation_normalizer_keeps_unknown_meaning_as_other() -> None:
    normalizer = ExactRelationNormalizer(["associated_with", "OTHER"])

    mapping = normalizer.normalize("was observed before")

    assert mapping.canonical_relation == "OTHER"
    assert mapping.confidence == 0.0
