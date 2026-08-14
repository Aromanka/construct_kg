from medical_kg.models.enums import StringEnum


class JobStatus(StringEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ProcessingStage(StringEnum):
    EXTRACTION = "extract"
    ENTITY_CANONICALIZATION = "entity_canonicalize"
    RELATION_CANONICALIZATION = "relation_canonicalize"
    GOLD_VALIDATION = "gold_validate"
