from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from medical_kg.models.entity import EntityMentionOutput


class Qualifiers(BaseModel):
    model_config = ConfigDict(extra="allow")

    species: str | None = None
    population: str | None = None
    age: str | None = None
    sex: str | None = None
    disease_state: str | None = None
    experimental_model: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    timepoint: str | None = None
    measurement_method: str | None = None
    direction: str | None = None
    effect_size: str | None = None
    statistical_significance: str | None = None
    condition: str | None = None
    tissue: str | None = None
    cell_type: str | None = None
    study_type: str | None = None


class AssertionOutput(BaseModel):
    subject: EntityMentionOutput
    object: EntityMentionOutput
    detailed_relation: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    qualifiers: Qualifiers = Field(default_factory=Qualifiers)
    negated: bool = False
    speculative: bool = False
    llm_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("detailed_relation", "evidence_text")
    @classmethod
    def reject_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must contain non-whitespace text")
        return value


class ExtractionOutput(BaseModel):
    assertions: list[AssertionOutput] = Field(default_factory=list)

    def as_json_value(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

