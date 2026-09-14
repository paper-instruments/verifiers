from __future__ import annotations

from collections.abc import Mapping

import aiohttp
import pytest

from verifiers.v1.clients import Client, ModelContext
from verifiers.v1.dialects import Dialect
from verifiers.v1.graph import PendingTurn
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace, TraceTask
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    Response,
    Sampling,
    SamplingConfig,
    ToolMessage,
    TurnTokens,
)


class CanonicalizingClient(Client):
    def __init__(self) -> None:
        self.turns: list[PendingTurn] = []
        self.raw_tool_orders: list[list[str]] = []

    def canonicalize_prompt(self, prompt: Messages) -> Messages:
        canonical = list(prompt)
        start = 0
        while start < len(canonical):
            if not isinstance(canonical[start], ToolMessage):
                start += 1
                continue
            end = start + 1
            while end < len(canonical) and isinstance(canonical[end], ToolMessage):
                end += 1
            canonical[start:end] = sorted(
                canonical[start:end], key=lambda message: message.tool_call_id
            )
            start = end
        return canonical

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        assert turn is not None
        self.turns.append(turn)
        self.raw_tool_orders.append(
            [
                message["tool_call_id"]
                for message in body["messages"]
                if message["role"] == "tool"
            ]
        )
        prompt_ids = [
            101
            if isinstance(message, AssistantMessage) and message.content == "answer-1"
            else 10 + index
            for index, message in enumerate(turn.prompt)
        ]
        completion_id = 100 + len(self.turns)
        response = Response(
            id=f"response-{len(self.turns)}",
            created=0,
            model=model,
            message=AssistantMessage(content=f"answer-{len(self.turns)}"),
            finish_reason="stop",
            usage=None,
            tokens=TurnTokens(
                prompt_ids=prompt_ids,
                completion_ids=[completion_id],
                completion_logprobs=[-0.1],
                message_spans=[(index, index + 1) for index in range(len(turn.prompt))],
            ),
        )
        response.raw = {
            "id": response.id,
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.message.content,
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        return response


class FailingCanonicalizingClient(CanonicalizingClient):
    def canonicalize_prompt(self, prompt: Messages) -> Messages:
        raise ValueError("unsupported tool-result ordering")


@pytest.mark.asyncio
async def test_interception_canonicalizes_before_prepare_and_commit() -> None:
    client = CanonicalizingClient()
    trace = Trace(
        task=TraceTask(
            type="Task",
            data=TaskData(idx=0, name="canonical-order", prompt="inspect"),
        )
    )
    session = RolloutSession(
        ctx=ModelContext(model="test-model", client=client, sampling=Sampling()),
        trace=trace,
    )
    initial_messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "a", "arguments": "{}"},
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "b", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-b", "content": "result-b"},
        {"role": "tool", "tool_call_id": "call-a", "content": "result-a"},
    ]

    async with InterceptionServer() as server:
        async with server.acquire(session) as (base_url, secret):
            headers = {"Authorization": f"Bearer {secret}"}
            async with aiohttp.ClientSession(headers=headers) as http:
                first_http = await http.post(
                    f"{base_url}/v1/chat/completions",
                    json={"model": "ignored", "messages": initial_messages},
                )
                assert first_http.status == 200
                first = await first_http.json()

                second_messages = [
                    *initial_messages,
                    first["choices"][0]["message"],
                    {"role": "user", "content": "continue"},
                ]
                second_http = await http.post(
                    f"{base_url}/v1/chat/completions",
                    json={"model": "ignored", "messages": second_messages},
                )
                assert second_http.status == 200
                await second_http.read()

    assert client.raw_tool_orders == [["call-b", "call-a"]] * 2
    assert [message.tool_call_id for message in client.turns[0].prompt[2:4]] == [
        "call-a",
        "call-b",
    ]
    assert client.turns[1].prefix_node_ids == list(range(5))
    branch = trace.branches[0]
    assert [message.tool_call_id for message in branch.messages[2:4]] == [
        "call-a",
        "call-b",
    ]
    assert branch.token_ids == [10, 11, 12, 13, 101, 15, 102]


@pytest.mark.asyncio
async def test_interception_reports_prompt_canonicalization_failure() -> None:
    client = FailingCanonicalizingClient()
    trace = Trace(
        task=TraceTask(
            type="Task",
            data=TaskData(idx=0, name="canonical-order-error", prompt="inspect"),
        )
    )
    session = RolloutSession(
        ctx=ModelContext(model="test-model", client=client, sampling=Sampling()),
        trace=trace,
    )

    async with InterceptionServer() as server:
        async with server.acquire(session) as (base_url, secret):
            headers = {"Authorization": f"Bearer {secret}"}
            async with aiohttp.ClientSession(headers=headers) as http:
                response = await http.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": "ignored",
                        "messages": [{"role": "user", "content": "inspect"}],
                    },
                )
                payload = await response.json()

    assert response.status == 502
    assert payload["error"]["message"] == "unsupported tool-result ordering"
    assert client.turns == []
    assert trace.nodes == []
    assert len(trace.calls) == 1
    assert trace.calls[0].error is not None
    assert trace.calls[0].error.type == "ValueError"


def test_client_prompt_canonicalization_defaults_to_identity() -> None:
    prompt: Messages = []

    assert Client.canonicalize_prompt(CanonicalizingClient(), prompt) is prompt
