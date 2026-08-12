from medical_kg.db.models import Base


def test_required_postgresql_tables_are_declared() -> None:
    expected = {
        "documents",
        "document_revisions",
        "processing_jobs",
        "extraction_runs",
        "entity_mentions",
        "entities",
        "entity_aliases",
        "entity_external_ids",
        "raw_assertions",
        "assertions",
        "relation_types",
        "invalid_records",
    }
    assert expected <= set(Base.metadata.tables)
