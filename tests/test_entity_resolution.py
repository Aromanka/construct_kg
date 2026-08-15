from medical_kg.silver.entity_resolution import (
    CandidateRetriever,
    ConservativeEntityResolver,
    EntityCandidate,
)


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


def test_resolves_medical_abbreviation_without_crossing_entity_types() -> None:
    resolver = ConservativeEntityResolver()
    candidates = [
        EntityCandidate("ENT_DISEASE", "type 2 diabetes mellitus", "DISEASE"),
        EntityCandidate("ENT_GENE", "type 2 diabetes mellitus", "GENE"),
    ]

    assert resolver.resolve("T2DM", "DISEASE", candidates) == "ENT_DISEASE"
    assert resolver.resolve("T2DM", "PROTEIN", candidates) is None


def test_lexical_similarity_only_retrieves_a_candidate() -> None:
    retriever = CandidateRetriever()
    candidates = [
        EntityCandidate("ENT_T2D", "type 2 diabetes", "DISEASE"),
        EntityCandidate("ENT_T1D", "type 1 diabetes", "DISEASE"),
    ]

    result = retriever.retrieve("diabetes mellitus", "DISEASE", candidates)

    assert {item.candidate.entity_id for item in result} == {"ENT_T1D", "ENT_T2D"}
