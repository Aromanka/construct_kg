from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from medical_kg.models.assertion import ExtractionOutput
from medical_kg.silver.schemas import (
    EntityCanonicalizationDecision,
    RelationCanonicalizationDecision,
)


@dataclass(frozen=True)
class LLMResponse:
    output: ExtractionOutput
    raw_output: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    max_request_count = 1

    @abstractmethod
    async def extract_document(
        self, *, system_prompt: str, user_prompt: str, temperature: float
    ) -> LLMResponse:
        """Extract validated assertions from the supplied document text or text chunk."""

    async def canonicalize_entity(
        self, *, system_prompt: str, user_prompt: str, temperature: float
    ) -> EntityCanonicalizationDecision:
        raise NotImplementedError("Entity canonicalization is not supported by this client")

    async def canonicalize_relation(
        self, *, system_prompt: str, user_prompt: str, temperature: float
    ) -> RelationCanonicalizationDecision:
        raise NotImplementedError("Relation canonicalization is not supported by this client")

    async def aclose(self) -> None:
        """Release provider resources when the client owns network connections."""
        return None
