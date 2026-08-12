from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from medical_kg.models.assertion import ExtractionOutput


@dataclass(frozen=True)
class LLMResponse:
    output: ExtractionOutput
    raw_output: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    async def extract_document(
        self, *, system_prompt: str, user_prompt: str, temperature: float
    ) -> LLMResponse:
        """Extract validated assertions from a complete document."""

    async def canonicalize_entity(self, **_: Any) -> Any:
        raise NotImplementedError("Entity canonicalization is a Phase II extension")

    async def canonicalize_relation(self, **_: Any) -> Any:
        raise NotImplementedError("Relation canonicalization is a Phase II extension")

