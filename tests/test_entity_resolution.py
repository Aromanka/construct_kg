from medical_kg.silver.entity_resolution import ConservativeEntityResolver, EntityCandidate


def test_resolves_one_exact_normalized_alias() -> None:
    resolver = ConservativeEntityResolver()
    result = resolver.resolve(
        "Type-II Diabetes",
        "DISEASE",
        [EntityCandidate("ENT_00000001", "type 2 diabetes", "DISEASE", ("type II diabetes",))],
    )
    assert result == "ENT_00000001"


def test_ambiguous_alias_is_not_merged() -> None:
    resolver = ConservativeEntityResolver()
    candidates = [
        EntityCandidate("ENT_00000001", "ABC", "GENE"),
        EntityCandidate("ENT_00000002", "ABC", "GENE"),
    ]
    assert resolver.resolve("ABC", "GENE", candidates) is None
