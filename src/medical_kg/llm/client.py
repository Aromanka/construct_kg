from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from medical_kg.config import AppSettings
from medical_kg.llm.base import LLMClient, LLMResponse
from medical_kg.models.assertion import ExtractionOutput


class OpenAIClient(LLMClient):
    """OpenAI and OpenAI-compatible structured-output client."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def extract_document(
        self, *, system_prompt: str, user_prompt: str, temperature: float
    ) -> LLMResponse:
        completion = await self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=ExtractionOutput,
        )
        message = completion.choices[0].message
        if message.refusal:
            raise ValueError(f"Model refused extraction: {message.refusal}")
        if message.parsed is None:
            raise ValueError("Model response did not contain valid structured extraction output")
        raw: dict[str, Any] = completion.model_dump(mode="json")
        usage = completion.usage
        return LLMResponse(
            output=message.parsed,
            raw_output=raw,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            metadata={"model": completion.model, "response_id": completion.id},
        )


def create_llm_client(settings: AppSettings) -> LLMClient:
    provider = settings.llm.provider.lower()
    if provider not in {"openai", "compatible"}:
        raise ValueError(f"Unsupported LLM provider {provider!r}; use 'openai' or 'compatible'")
    return OpenAIClient(
        api_key=settings.llm.api_key.get_secret_value(),
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
    )

