import json
from pathlib import Path

import httpx
import pytest

from medical_kg.llm.client import CompatibleAPIClient


@pytest.mark.asyncio
async def test_compatible_client_extracts_deepseek_json_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-chat"
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "deepseek-chat",
                "choices": [
                    {
                        "message": {
                            "content": "```json\n{\"assertions\": []}\n```"
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    client = CompatibleAPIClient(
        api_key="test-key",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.extract_document(
            system_prompt="Extract knowledge.",
            user_prompt="Document text.",
            temperature=0.0,
        )
    finally:
        await client.aclose()

    assert response.output.assertions == []
    assert response.input_tokens == 12
    assert response.output_tokens == 4


@pytest.mark.asyncio
async def test_compatible_client_does_not_read_stale_ssl_cert_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))

    client = CompatibleAPIClient(
        api_key="test-key",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"assertions": []}'}}]},
            )
        ),
    )
    await client.aclose()
