from medical_kg.silver.entity_resolution import (
    CandidateRetriever,
    ConservativeEntityResolver,
    EntityCandidate,
    IndexedEntityResolver,
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


def test_resolves_same_normalized_token_set_regardless_of_word_order() -> None:
    resolver = ConservativeEntityResolver()
    candidates = [
        EntityCandidate("ENT_DKD", "diabetic kidney disease", "DISEASE"),
        EntityCandidate("ENT_GENE", "diabetic kidney disease", "GENE"),
    ]

    decision = resolver.resolve_decision("Kidney-disease diabetic", "DISEASE", candidates)

    assert decision.entity_id == "ENT_DKD"
    assert decision.method == "token_set"
    assert resolver.resolve("kidney disease diabetic", "GENE", candidates) == "ENT_GENE"
    assert resolver.resolve("kidney disease diabetic", "DRUG", candidates) is None


def test_ambiguous_token_set_is_not_merged() -> None:
    resolver = ConservativeEntityResolver()
    candidates = [
        EntityCandidate("ENT_1", "alpha beta", "GENE"),
        EntityCandidate("ENT_2", "beta alpha", "GENE"),
    ]

    assert resolver.resolve("alpha beta beta", "GENE", candidates) is None


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


def test_indexed_resolver_matches_deterministic_resolution_rules() -> None:
    candidates = [
        EntityCandidate(
            "ENT_DISEASE",
            "type 2 diabetes mellitus",
            "DISEASE",
            ("type II diabetes",),
        ),
        EntityCandidate("ENT_GENE", "type 2 diabetes mellitus", "GENE"),
    ]
    resolver = IndexedEntityResolver(candidates)

    assert resolver.resolve_decision("Type-II Diabetes", "DISEASE").entity_id == "ENT_DISEASE"
    assert resolver.resolve_decision("T2DM", "DISEASE").entity_id == "ENT_DISEASE"
    assert resolver.resolve_decision("T2DM", "PROTEIN").entity_id is None


def test_indexed_resolver_matches_same_token_set_regardless_of_word_order() -> None:
    resolver = IndexedEntityResolver(
        [EntityCandidate("ENT_DKD", "diabetic kidney disease", "DISEASE")]
    )

    decision = resolver.resolve_decision("kidney disease diabetic", "DISEASE")

    assert decision.entity_id == "ENT_DKD"
    assert decision.method == "token_set"


def test_indexed_resolver_updates_when_entities_and_aliases_are_added() -> None:
    resolver = IndexedEntityResolver()
    resolver.add_candidate(EntityCandidate("ENT_NEW", "diabetic kidney disease", "DISEASE"))
    resolver.add_alias("ENT_NEW", "DISEASE", "DKD")

    assert resolver.resolve_decision("DKD", "DISEASE").entity_id == "ENT_NEW"
    assert resolver.candidates("DISEASE")[0].aliases == ("DKD",)
