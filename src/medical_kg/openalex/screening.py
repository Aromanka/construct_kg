from __future__ import annotations

import json
import ssl
from collections.abc import Sequence
from typing import Protocol

import httpx

from medical_kg.config import AppSettings
from medical_kg.openalex.models import OpenAlexWork


class WorkScreener(Protocol):
    async def screen(self, works: Sequence[OpenAlexWork], instruction: str) -> set[int]:
        """Return zero-based indexes selected from one numbered batch."""

    async def aclose(self) -> None: ...


class CompatibleWorkScreener:
    """Batch title/abstract screening through the configured compatible LLM API."""

    def __init__(self, settings: AppSettings) -> None:
        if not settings.llm.base_url:
            raise ValueError("llm.base_url is required for OpenAlex LLM screening")
        self.model = settings.llm.model
        self.temperature = settings.llm.temperature
        self.client = httpx.AsyncClient(
            base_url=settings.llm.base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {settings.llm.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=settings.llm.timeout,
            verify=ssl.create_default_context(),
        )

    async def screen(self, works: Sequence[OpenAlexWork], instruction: str) -> set[int]:
        articles = "\n\n".join(
            f"[{index}]\nTITLE: {work.title}\nABSTRACT: {work.abstract or '[NO ABSTRACT]'}"
            for index, work in enumerate(works, start=1)
        )
        response = await self.client.post(
            "chat/completions",
            json={
                "model": self.model,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You screen literature relevance. Apply the user's instruction "
                            "only to the numbered titles and abstracts. Return one JSON object "
                            "with selected_numbers, an array of relevant 1-based integers. "
                            "Do not invent numbers or include explanations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"SCREENING INSTRUCTION:\n{instruction}\n\nARTICLES:\n{articles}\n\n"
                            '{"selected_numbers":[1,2]}'
                        ),
                    },
                ],
            },
        )
        response.raise_for_status()
        raw = response.json()
        try:
            content = raw["choices"][0]["message"]["content"]
            payload = json.loads(content)
            numbers = payload["selected_numbers"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("LLM screening response is not valid selected_numbers JSON") from error
        if not isinstance(numbers, list) or any(
            isinstance(number, bool) or not isinstance(number, int) for number in numbers
        ):
            raise ValueError("selected_numbers must be an array of integers")
        maximum = len(works)
        if any(number < 1 or number > maximum for number in numbers):
            raise ValueError(f"LLM selected a number outside 1..{maximum}")
        return {number - 1 for number in numbers}

    async def aclose(self) -> None:
        await self.client.aclose()
