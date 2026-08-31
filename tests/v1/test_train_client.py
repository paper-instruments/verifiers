from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APITimeoutError
from pydantic import ValidationError
from renderers import ParsedToolCall

from verifiers.v1.clients.base import DEFAULT_TIMEOUT
from verifiers.v1.clients.qwen38 import (
    QWEN38_MODEL_ID,
    is_qwen38_model,
)
from verifiers.v1.clients.train import (
    ElasticRendererPool,
    TrainClient,
    response_from_generate,
)
from verifiers.v1.configs.client import TrainClientConfig
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.errors import ProviderError
from verifiers.v1.types import SamplingConfig

QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def test_synthetic_tool_call_ids_are_unique_across_responses():
    first = response_from_generate(_generate_result("request-a"), "model")
    second = response_from_generate(_generate_result("request-b"), "model")

    assert first.message.tool_calls is not None
    assert second.message.tool_calls is not None
    assert first.message.tool_calls[0].id == "call_request-a_0"
    assert second.message.tool_calls[0].id == "call_request-b_0"


def test_renderer_native_tool_call_id_is_preserved():
    response = response_from_generate(
        _generate_result("request-a", "functions.lookup:0"),
        "model",
    )

    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].id == "functions.lookup:0"


@pytest.mark.parametrize(
    "model_name,expected",
    [
        (QWEN38_MODEL_ID, True),
        (
            "/model_cache/hub/models--Qwen--Qwen3.8-27B/snapshots/" + QWEN38_REVISION,
            True,
        ),
        ("Qwen/Qwen3.6-27B", False),
    ],
    ids=["canonical", "snapshot", "qwen36"],
)
def test_is_qwen38_model_matches_only_canonical_or_snapshot(model_name, expected):
    assert is_qwen38_model(model_name) is expected


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_train_client_rejects_invalid_inference_timeout(timeout):
    with pytest.raises(ValidationError, match="inference_read_timeout_seconds"):
        TrainClientConfig(inference_read_timeout_seconds=timeout)


@pytest.mark.parametrize(
    ("configured_timeout", "expected_timeout"),
    [(None, DEFAULT_TIMEOUT.read), (0.01, 0.01)],
)
async def test_train_client_timeout_is_configured_and_attributed(
    monkeypatch, configured_timeout, expected_timeout
):
    client = TrainClient(
        TrainClientConfig(
            base_url="http://router:8000/v1",
            inference_read_timeout_seconds=configured_timeout,
        )
    )
    assert client.client.timeout.read == expected_timeout
    if configured_timeout is None:
        assert client.client.timeout is DEFAULT_TIMEOUT

    slot = SimpleNamespace(
        renderer=object(),
        run=AsyncMock(
            return_value=SimpleNamespace(
                token_ids=[1],
                multi_modal_data=None,
                prompt_attribution=None,
            )
        ),
    )

    @asynccontextmanager
    async def acquire(_self):
        yield slot

    async def time_out(**_kwargs):
        raise APITimeoutError(request=httpx.Request("POST", "http://router"))

    monkeypatch.setattr(ElasticRendererPool, "acquire", acquire)
    monkeypatch.setattr("renderers.client.generate", time_out)

    try:
        with pytest.raises(ProviderError) as exc_info:
            await client.get_response(
                ChatDialect(),
                {"messages": [{"role": "user", "content": "hello"}]},
                "model",
                SamplingConfig(),
                session_id="trace-123",
            )
    finally:
        await client.close()

    error = exc_info.value
    assert error.status_code == 504
    assert "owner=verifiers.train_client" in str(error)
    assert f"configured_read_timeout_seconds={configured_timeout}" in str(error)
    assert f"effective_read_timeout_seconds={expected_timeout}" in str(error)
    assert "elapsed_seconds=" in str(error)
    assert "http_status=504" in str(error)
    assert "session_id=trace-123" in str(error)
    assert "endpoint=http://router:8000/v1" in str(error)


def _generate_result(request_id: str, tool_call_id: str | None = None) -> dict:
    return {
        "request_id": request_id,
        "finish_reason": "tool_calls",
        "prompt_ids": [1],
        "completion_ids": [2],
        "completion_logprobs": [-0.1],
        "tool_calls": [
            ParsedToolCall(
                raw="raw",
                name="lookup",
                arguments={},
                id=tool_call_id,
            )
        ],
    }
