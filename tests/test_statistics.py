from medical_kg.utils.statistics import (
    assemble_knowledge_statistics,
    compute_graph_quality,
)


def test_assemble_knowledge_statistics_summarizes_committed_work() -> None:
    result = assemble_knowledge_statistics(
        totals={
            "documents": 4,
            "document_revisions": 4,
            "processing_jobs": 12,
            "extraction_runs": 5,
            "extracted_documents": 2,
            "entity_mentions": 20,
            "unique_mention_texts": 8,
            "raw_assertions": 10,
            "evidence_validated": 9,
            "assertion_documents": 2,
            "entities": 0,
            "assertions": 0,
            "relation_types": 15,
            "invalid_records": 1,
        },
        source_counts=[("guidelines", 4)],
        job_counts=[
            ("extract:clinical", "v1", "RUNNING", 1),
            ("extract:general", "v1", "SUCCESS", 4),
            ("extract:molecular", "v1", "PENDING", 7),
        ],
        pass_counts=[("clinical", 1), ("general", 4)],
        entity_type_counts=[("DISEASE", 12), ("DRUG", 8)],
    )

    assert result["documents"] == {
        "total": 4,
        "revisions": 4,
        "by_source_type": {"guidelines": 4},
    }
    assert result["processing_jobs"]["by_status"] == {
        "PENDING": 7,
        "RUNNING": 1,
        "SUCCESS": 4,
    }
    assert result["bronze_knowledge"]["raw_assertions"] == 10
    assert result["bronze_knowledge"]["evidence_not_validated"] == 1
    assert result["quality"]["evidence_validation_rate"] == 0.9
    assert result["canonical_knowledge"]["entities"] == 0


def test_assemble_knowledge_statistics_uses_null_rate_when_no_assertions() -> None:
    result = assemble_knowledge_statistics(
        totals={
            "documents": 0,
            "document_revisions": 0,
            "processing_jobs": 0,
            "extraction_runs": 0,
            "extracted_documents": 0,
            "entity_mentions": 0,
            "unique_mention_texts": 0,
            "raw_assertions": 0,
            "evidence_validated": 0,
            "assertion_documents": 0,
            "entities": 0,
            "assertions": 0,
            "relation_types": 0,
            "invalid_records": 0,
        },
        source_counts=[],
        job_counts=[],
        pass_counts=[],
        entity_type_counts=[],
    )

    assert result["quality"]["evidence_validation_rate"] is None


def test_compute_graph_quality_reports_connectivity_and_reuse() -> None:
    result = compute_graph_quality(
        edges=[
            ("t2d", "metformin", "treats"),
            ("t2d", "obesity", "associated_with"),
            ("isolated-a", "isolated-b", "OTHER"),
        ],
        mention_count=12,
        entity_count=5,
        entity_documents=[
            ("t2d", "doc-1"),
            ("t2d", "doc-2"),
            ("metformin", "doc-1"),
        ],
        evidence_count=5,
    )

    assert result == {
        "singleton_edge_ratio": 0.3333,
        "largest_connected_component_ratio": 0.6,
        "canonical_compression_ratio": 2.4,
        "cross_document_reuse_rate": 0.2,
        "relation_other_ratio": 0.3333,
        "duplicate_canonical_assertion_ratio": 0.4,
    }
