from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from medical_kg.models.enums import StringEnum


class EntityType(StringEnum):
    DISEASE = "DISEASE"
    PHENOTYPE = "PHENOTYPE"
    SYMPTOM = "SYMPTOM"
    DRUG = "DRUG"
    COMPOUND = "COMPOUND"
    METABOLITE = "METABOLITE"
    GENE = "GENE"
    PROTEIN = "PROTEIN"
    PATHWAY = "PATHWAY"
    CELL = "CELL"
    TISSUE = "TISSUE"
    ORGAN = "ORGAN"
    BIOMARKER = "BIOMARKER"
    LAB_MEASUREMENT = "LAB_MEASUREMENT"
    PHYSIOLOGICAL_PROCESS = "PHYSIOLOGICAL_PROCESS"
    BIOLOGICAL_PROCESS = "BIOLOGICAL_PROCESS"
    TREATMENT = "TREATMENT"
    INTERVENTION = "INTERVENTION"
    PROCEDURE = "PROCEDURE"
    DIET = "DIET"
    NUTRIENT = "NUTRIENT"
    FOOD = "FOOD"
    EXERCISE = "EXERCISE"
    BEHAVIOR = "BEHAVIOR"
    LIFESTYLE_FACTOR = "LIFESTYLE_FACTOR"
    ENVIRONMENTAL_EXPOSURE = "ENVIRONMENTAL_EXPOSURE"
    RISK_FACTOR = "RISK_FACTOR"
    POPULATION = "POPULATION"
    CLINICAL_OUTCOME = "CLINICAL_OUTCOME"
    OTHER = "OTHER"


class EntityMentionOutput(BaseModel):
    mention: str = Field(min_length=1)
    entity_type: EntityType
    entity_type_detail: str | None = None

    @model_validator(mode="after")
    def require_other_detail(self) -> EntityMentionOutput:
        if self.entity_type == EntityType.OTHER and not self.entity_type_detail:
            raise ValueError("entity_type_detail is required when entity_type is OTHER")
        return self
