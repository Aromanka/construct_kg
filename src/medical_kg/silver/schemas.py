from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EntityCanonicalizationDecision(BaseModel):
    decision: Literal["MATCH", "NEW"]
    entity_id: str | None = None
    canonical_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_decision_payload(self) -> EntityCanonicalizationDecision:
        if self.decision == "MATCH" and not self.entity_id:
            raise ValueError("entity_id is required for MATCH")
        if self.decision == "NEW" and not self.canonical_name:
            raise ValueError("canonical_name is required for NEW")
        return self


class RelationCanonicalizationDecision(BaseModel):
    canonical_relation: str
    confidence: float = Field(ge=0.0, le=1.0)
