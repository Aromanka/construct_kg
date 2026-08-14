from __future__ import annotations

import json
import ssl
from typing import Any

import httpx

from medical_kg.config import AppSettings
from medical_kg.llm.base import LLMClient, LLMResponse
from medical_kg.models.assertion import ExtractionOutput


class CompatibleAPIClient(LLMClient):
    """Lightweight client for DeepSeek and similar OpenAI-compatible APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            # An explicit system context keeps TLS verification enabled while preventing HTTPX
            # from replacing it with a stale SSL_CERT_FILE environment override.
            verify=ssl.create_default_context(),
            transport=transport,
        )

    async def extract_document(
        self, *, system_prompt: str, user_prompt: str, temperature: float
    ) -> LLMResponse:
        schema = json.dumps(
            ExtractionOutput.model_json_schema(), ensure_ascii=False, separators=(",", ":")
        )
        response = await self.client.post(
            "chat/completions",
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": f"{system_prompt}\nReturn only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": f"{user_prompt}\n\nJSON SCHEMA:\n{schema}",
                    },
                ],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict):
            raise ValueError("Compatible API response must be a JSON object")
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Compatible API response does not contain message content") from error
        if not isinstance(content, str):
            raise ValueError("Compatible API message content must be a string")

        output = ExtractionOutput.model_validate(self._decode_json_content(content))
        usage = raw.get("usage") or {}
        return LLMResponse(
            output=output,
            raw_output=raw,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            metadata={"model": raw.get("model", self.model), "response_id": raw.get("id")},
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    @staticmethod
    def _decode_json_content(content: str) -> Any:
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[-1].strip() == "```":
                lines = lines[1:-1]
                content = "\n".join(lines).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Compatible API returned invalid JSON content") from error


def create_llm_client(settings: AppSettings) -> LLMClient:
    provider = settings.llm.provider.lower()
    if provider not in {"compatible", "deepseek"}:
        raise ValueError(
            f"Unsupported LLM provider {provider!r}; use 'compatible' or 'deepseek'"
        )
    if not settings.llm.base_url:
        raise ValueError("llm.base_url is required for a compatible third-party provider")
    return CompatibleAPIClient(
        api_key=settings.llm.api_key.get_secret_value(),
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
    )
