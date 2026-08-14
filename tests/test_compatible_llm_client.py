import json
from pathlib import Path

import httpx
import pytest

from medical_kg.llm.client import CompatibleAPIClient, StructuredOutputValidationError


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


@pytest.mark.asyncio
async def test_compatible_client_normalizes_null_qualifiers_without_another_request() -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "assertions": [
                                        {
                                            "subject": {
                                                "mention": "diabetes",
                                                "entity_type": "DISEASE",
                                            },
                                            "object": {
                                                "mention": "metformin",
                                                "entity_type": "DRUG",
                                            },
                                            "detailed_relation": "is treated with",
                                            "evidence_text": "Diabetes is treated with metformin.",
                                            "qualifiers": None,
                                            "llm_confidence": 0.9,
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
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
            system_prompt="Extract knowledge.", user_prompt="Document.", temperature=0.0
        )
    finally:
        await client.aclose()

    assert requests == 1
    assert response.output.assertions[0].qualifiers.model_dump(exclude_none=True) == {}
    assert response.metadata["structured_output_corrected"] is False


@pytest.mark.asyncio
async def test_compatible_client_corrects_invalid_other_detail_once() -> None:
    requests: list[dict[str, object]] = []
    invalid = {
        "assertions": [
            {
                "subject": {"mention": "CGM system", "entity_type": "OTHER"},
                "object": {"mention": "glucose", "entity_type": "LAB_MEASUREMENT"},
                "detailed_relation": "measures",
                "evidence_text": "The CGM system measures glucose.",
                "qualifiers": {},
                "llm_confidence": 0.8,
            }
        ]
    }
    corrected = json.loads(json.dumps(invalid))
    corrected["assertions"][0]["subject"]["entity_type_detail"] = "monitoring device"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        content = invalid if len(requests) == 1 else corrected
        return httpx.Response(
            200,
            json={
                "id": f"response-{len(requests)}",
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {
                    "prompt_tokens": 10 * len(requests),
                    "completion_tokens": 2 * len(requests),
                },
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
            system_prompt="Extract knowledge.", user_prompt="Document.", temperature=0.0
        )
    finally:
        await client.aclose()

    assert len(requests) == 2
    assert "VALIDATION ERRORS" in requests[1]["messages"][-1]["content"]
    assert response.output.assertions[0].subject.entity_type_detail == "monitoring device"
    assert response.input_tokens == 30
    assert response.output_tokens == 6
    assert response.metadata["request_count"] == 2
    assert response.metadata["structured_output_corrected"] is True
    assert set(response.raw_output) == {
        "initial_response",
        "corrected_response",
        "initial_validation_errors",
    }


@pytest.mark.asyncio
async def test_compatible_client_keeps_other_detail_strict_after_correction() -> None:
    invalid = {
        "assertions": [
            {
                "subject": {"mention": "study", "entity_type": "OTHER"},
                "object": {"mention": "diabetes", "entity_type": "DISEASE"},
                "detailed_relation": "examines",
                "evidence_text": "The study examines diabetes.",
                "qualifiers": {},
                "llm_confidence": 0.7,
            }
        ]
    }
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(invalid)}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    client = CompatibleAPIClient(
        api_key="test-key",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(StructuredOutputValidationError) as captured:
            await client.extract_document(
                system_prompt="Extract knowledge.", user_prompt="Document.", temperature=0.0
            )
    finally:
        await client.aclose()

    assert requests == 2
    assert captured.value.request_count == 2
    assert captured.value.input_tokens == 6
    assert captured.value.output_tokens == 4
    assert "entity_type_detail is required" in str(captured.value)
