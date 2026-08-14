from __future__ import annotations

import copy
import json
import ssl
from typing import Any

import httpx
from pydantic import ValidationError

from medical_kg.config import AppSettings
from medical_kg.llm.base import LLMClient, LLMResponse
from medical_kg.models.assertion import ExtractionOutput


class StructuredOutputValidationError(ValueError):
    """The provider output was still invalid after one guided correction."""

    def __init__(
        self,
        validation_error: ValidationError,
        *,
        raw_outputs: list[dict[str, Any]],
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.validation_error = validation_error
        self.raw_outputs = raw_outputs
        self.request_count = len(raw_outputs)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        super().__init__(
            "Structured output remained invalid after one correction request:\n"
            f"{validation_error}"
        )


class CompatibleAPIClient(LLMClient):
    """Lightweight client for DeepSeek and similar OpenAI-compatible APIs."""

    max_request_count = 2

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 120.0,
        max_connections: int = 100,
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
            limits=httpx.Limits(
                max_connections=max_connections + max(10, max_connections // 5),
                max_keepalive_connections=max_connections,
                keepalive_expiry=60.0,
            ),
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
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\nReturn only valid JSON.",
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n\nJSON SCHEMA:\n{schema}",
            },
        ]
        raw = await self._completion(messages=messages, temperature=temperature)
        payload = self._normalize_output(self._decode_response_content(raw))
        input_tokens, output_tokens = self._usage(raw)
        try:
            output = ExtractionOutput.model_validate(payload)
        except ValidationError as initial_error:
            correction = self._correction_message(initial_error)
            corrected_raw = await self._completion(
                messages=[
                    *messages,
                    {
                        "role": "assistant",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                    {"role": "user", "content": correction},
                ],
                temperature=temperature,
            )
            corrected_payload = self._normalize_output(
                self._decode_response_content(corrected_raw)
            )
            corrected_input, corrected_output = self._usage(corrected_raw)
            input_tokens += corrected_input
            output_tokens += corrected_output
            try:
                output = ExtractionOutput.model_validate(corrected_payload)
            except ValidationError as final_error:
                raise StructuredOutputValidationError(
                    final_error,
                    raw_outputs=[raw, corrected_raw],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ) from final_error
            return LLMResponse(
                output=output,
                raw_output={
                    "initial_response": raw,
                    "corrected_response": corrected_raw,
                    "initial_validation_errors": initial_error.errors(
                        include_url=False, include_context=False, include_input=False
                    ),
                },
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={
                    "model": corrected_raw.get("model", self.model),
                    "response_id": corrected_raw.get("id"),
                    "request_count": 2,
                    "structured_output_corrected": True,
                },
            )

        return LLMResponse(
            output=output,
            raw_output=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "model": raw.get("model", self.model),
                "response_id": raw.get("id"),
                "request_count": 1,
                "structured_output_corrected": False,
            },
        )

    async def _completion(
        self, *, messages: list[dict[str, str]], temperature: float
    ) -> dict[str, Any]:
        response = await self.client.post(
            "chat/completions",
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict):
            raise ValueError("Compatible API response must be a JSON object")
        return raw

    async def aclose(self) -> None:
        await self.client.aclose()

    @classmethod
    def _decode_response_content(cls, raw: dict[str, Any]) -> Any:
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Compatible API response does not contain message content") from error
        if not isinstance(content, str):
            raise ValueError("Compatible API message content must be a string")
        return cls._decode_json_content(content)

    @staticmethod
    def _normalize_output(payload: Any) -> Any:
        """Apply only semantics-preserving repairs before strict validation."""
        normalized = copy.deepcopy(payload)
        if not isinstance(normalized, dict):
            return normalized
        assertions = normalized.get("assertions")
        if not isinstance(assertions, list):
            return normalized
        for assertion in assertions:
            if isinstance(assertion, dict) and assertion.get("qualifiers") is None:
                assertion["qualifiers"] = {}
        return normalized

    @staticmethod
    def _correction_message(error: ValidationError) -> str:
        errors = error.errors(include_url=False, include_context=False, include_input=False)
        return (
            "Your previous JSON failed schema validation. Return the complete corrected JSON "
            "object, not an explanation. Preserve supported assertions and exact evidence. "
            "When entity_type is OTHER, entity_type_detail must be a non-empty precise semantic "
            "category. Prefer a listed specific entity_type when one applies. qualifiers must "
            "always be a JSON object; use {} when there are no qualifiers and never use null.\n\n"
            f"VALIDATION ERRORS:\n{json.dumps(errors, ensure_ascii=False)}"
        )

    @staticmethod
    def _usage(raw: dict[str, Any]) -> tuple[int, int]:
        usage = raw.get("usage") or {}
        return (
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )

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
        max_connections=settings.processing.api_concurrency,
    )
