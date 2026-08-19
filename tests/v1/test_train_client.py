import pytest
from renderers import ParsedToolCall
from renderers.configs import Qwen36RendererConfig

from verifiers.v1.clients.qwen38 import (
    QWEN38_MODEL_ID,
    Qwen38Renderer,
    is_qwen38_model,
)
from verifiers.v1.clients.train import TrainClient, response_from_generate

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


def test_train_client_uses_qwen38_renderer_pool():
    config = Qwen36RendererConfig(enable_thinking=True, preserve_thinking=False)
    client = TrainClient(openai=object(), config=config)

    pool = client._renderer_pool(
        QWEN38_MODEL_ID,
        chat_template_kwargs={"reasoning_effort": "low"},
    )

    assert pool.renderer_cls is Qwen38Renderer
    with pool.checkout() as renderer:
        assert renderer.reasoning_effort == "low"
        assert renderer.config.preserve_thinking is False


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
