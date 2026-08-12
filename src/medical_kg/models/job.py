from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ProcessingStage(StrEnum):
    EXTRACTION = "extract"
    ENTITY_CANONICALIZATION = "entity_canonicalize"
    RELATION_CANONICALIZATION = "relation_canonicalize"
    GOLD_VALIDATION = "gold_validate"

