from renderers import ParsedToolCall

from verifiers.v1.clients.train import response_from_generate


def test_synthetic_tool_call_ids_are_unique_across_responses():
    first = response_from_generate(_generate_result("request-a"), "model")
    second = response_from_generate(_generate_result("request-b"), "model")

    assert first.message.tool_calls is not None
    assert second.message.tool_calls is not None
    assert first.message.tool_calls[0].id == "call_request-a_0"
    assert second.message.tool_calls[0].id == "call_request-b_0"


def test_renderer_native_tool_call_id_is_preserved():
    # Kimi emits this ID in sampled tokens, so Verifiers must not replace it.
    response = response_from_generate(
        _generate_result("request-a", "functions.lookup:0"),
        "model",
    )

    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].id == "functions.lookup:0"


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
