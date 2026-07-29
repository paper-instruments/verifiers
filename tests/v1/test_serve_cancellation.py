import asyncio

import msgpack
import pytest
import zmq

import verifiers.v1.serve.client as client_module
from verifiers.v1.clients.config import EvalClientConfig
from verifiers.v1.legacy import LegacyEnvServer
from verifiers.v1.loaders import env_config_type
from verifiers.v1.serve import (
    CancelRequest,
    CancelResponse,
    EnvClient,
    EnvServer,
    RunGroupRequest,
)
from verifiers.v1.serve.pool import EnvServerPool
from verifiers.v1.types import SamplingConfig


class CancellationState:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.cleanup_allowed = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def block(self) -> None:
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.cleanup_started.set()
            await self.cleanup_allowed.wait()
            self.cleaned.set()


class FakeDealer:
    def __init__(self) -> None:
        self.sent: list[list[bytes]] = []
        self.reply: list[zmq.Frame] | None = None

    async def send_multipart(self, frames: list[bytes]) -> None:
        self.sent.append(frames)

    async def recv_multipart(self, *, copy: bool) -> list[zmq.Frame]:
        assert not copy
        assert self.reply is not None
        return self.reply


@pytest.mark.parametrize("method", ["run", "run_group"])
async def test_client_cancellation_waits_for_remote_cleanup(method):
    state = CancellationState()
    server = _blocking_server(state, method)
    await _assert_client_cancellation_waits_for_cleanup(
        server, state, method, legacy=False
    )


@pytest.mark.parametrize("method", ["run", "run_group"])
async def test_legacy_client_cancellation_waits_for_remote_cleanup(method):
    state = CancellationState()
    server = _blocking_legacy_server(state, method)
    await _assert_client_cancellation_waits_for_cleanup(
        server, state, method, legacy=True
    )


async def test_client_cancellation_is_bounded_when_server_is_gone(monkeypatch):
    monkeypatch.setattr(client_module, "_REMOTE_CANCEL_TIMEOUT", 0.05)
    state = CancellationState()
    server = _blocking_server(state, "run")
    server_task = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    request_task = asyncio.create_task(_run(client, "run", legacy=False))
    try:
        await asyncio.wait_for(state.started.wait(), 2)
        state.cleanup_allowed.set()
        server_task.cancel()
        await asyncio.wait_for(server_task, 2)

        request_task.cancel("caller stopped")
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(request_task, 1)
        assert raised.value.args == ("caller stopped",)
        assert client._pending == {}
    finally:
        state.cleanup_allowed.set()
        await client.close()
        if not server_task.done():
            server_task.cancel()
            await asyncio.wait_for(server_task, 2)


async def test_cancel_unknown_target_is_a_successful_noop():
    state = CancellationState()
    server = _blocking_server(state, "run")
    server_task = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    try:
        response = await asyncio.wait_for(client.cancel("missing"), 2)
        assert response.success
        assert not response.cancelled
    finally:
        await client.close()
        server_task.cancel()
        await asyncio.wait_for(server_task, 2)


async def test_pool_routes_cancel_to_owner_and_releases_capacity():
    pool = EnvServerPool(
        server_kwargs={},
        max_workers=2,
        address="tcp://127.0.0.1:0",
        legacy=False,
        elastic=False,
    )
    dealers = [FakeDealer(), FakeDealer()]
    workers = [{"dealer": dealer, "active": 0} for dealer in dealers]
    pool.workers = workers
    responses: list[tuple[bytes, bytes, CancelResponse]] = []

    async def capture_response(
        client_id: bytes,
        request_id: bytes,
        response: CancelResponse,
    ) -> None:
        responses.append((client_id, request_id, response))

    pool._send_response = capture_response
    client_id = b"owner"
    target_id = b"target"
    cancel_id = b"cancel"
    payload = _cancel_payload(target_id)
    try:
        await pool._on_request(client_id, target_id, b"run", b"request")
        owner = workers[0]
        assert owner["active"] == 1
        assert pool._in_flight == 1

        await pool._on_cancel(b"other-client", b"rejected", payload)
        assert not responses[-1][2].cancelled
        assert target_id in pool._pending

        await pool._on_cancel(client_id, cancel_id, payload)
        assert dealers[0].sent[-1] == [cancel_id, b"cancel", payload]
        assert cancel_id in pool._pending

        response_data = msgpack.packb(
            CancelResponse(cancelled=True).model_dump(mode="json"),
            use_bin_type=True,
        )
        dealers[0].reply = [zmq.Frame(cancel_id), zmq.Frame(response_data)]
        await pool._on_reply(owner)

        assert pool._pending == {}
        assert pool._in_flight == 0
        assert owner["active"] == 0
        assert workers[1]["active"] == 0

        dealers[0].reply = [zmq.Frame(target_id), zmq.Frame(b"late")]
        await pool._on_reply(owner)
        assert pool._in_flight == 0
        assert owner["active"] == 0
    finally:
        pool.workers = []
        pool._shutdown()


async def test_pool_natural_completion_race_releases_capacity_once():
    pool = EnvServerPool(
        server_kwargs={},
        max_workers=1,
        address="tcp://127.0.0.1:0",
        legacy=True,
        elastic=False,
    )
    dealer = FakeDealer()
    worker = {"dealer": dealer, "active": 0}
    pool.workers = [worker]
    client_id = b"owner"
    target_id = b"target"
    cancel_id = b"cancel"
    try:
        await pool._on_request(
            client_id, target_id, b"run_group", _run_group_payload(n=3)
        )
        await pool._on_cancel(client_id, cancel_id, _cancel_payload(target_id))

        dealer.reply = [zmq.Frame(target_id), zmq.Frame(b"completed")]
        await pool._on_reply(worker)
        assert target_id not in pool._pending
        assert cancel_id in pool._pending
        assert pool._in_flight == 0
        assert worker["active"] == 0

        response_data = msgpack.packb(
            CancelResponse(cancelled=False).model_dump(mode="json"),
            use_bin_type=True,
        )
        dealer.reply = [zmq.Frame(cancel_id), zmq.Frame(response_data)]
        await pool._on_reply(worker)
        assert pool._pending == {}
        assert pool._in_flight == 0
        assert worker["active"] == 0
    finally:
        pool.workers = []
        pool._shutdown()


async def _run(client: EnvClient, method: str, *, legacy: bool):
    kwargs = {
        "client": EvalClientConfig(),
        "model": "model",
        "sampling": SamplingConfig(),
    }
    if method == "run_group":
        return await client.run_group(task_idx=0, n=2, **kwargs)
    if legacy:
        return await client.run(task_idx=0, **kwargs)
    return await client.run(task_data={}, **kwargs)


def _blocking_server(state: CancellationState, method: str) -> EnvServer:
    config_type = env_config_type("echo-v1")
    server = EnvServer(
        config=config_type(taskset={"id": "echo-v1"}),
        address="tcp://127.0.0.1:0",
    )

    async def block(*_args, **_kwargs):
        await state.block()

    if method == "run_group":
        server._run_group = block
    else:
        server._run = block
    return server


def _blocking_legacy_server(state: CancellationState, method: str) -> LegacyEnvServer:
    server = LegacyEnvServer(
        env_id="echo-v0",
        address="tcp://127.0.0.1:0",
    )

    async def block(*_args, **_kwargs):
        await state.block()

    if method == "run_group":
        server._run_group = block
    else:
        server._run = block
    return server


async def _assert_client_cancellation_waits_for_cleanup(
    server: EnvServer,
    state: CancellationState,
    method: str,
    *,
    legacy: bool,
) -> None:
    server_task = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    request_task = asyncio.create_task(_run(client, method, legacy=legacy))
    try:
        await asyncio.wait_for(state.started.wait(), 2)
        request_task.cancel()
        await asyncio.wait_for(state.cleanup_started.wait(), 2)
        await asyncio.sleep(0)
        assert not request_task.done()

        state.cleanup_allowed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request_task, 2)
        assert state.cleaned.is_set()
        assert client._pending == {}
        assert server._request_tasks == {}
    finally:
        state.cleanup_allowed.set()
        await client.close()
        server_task.cancel()
        await asyncio.wait_for(server_task, 2)


def _cancel_payload(target_request_id: bytes) -> bytes:
    request = CancelRequest(target_request_id=target_request_id.decode())
    return msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)


def _run_group_payload(n: int) -> bytes:
    request = RunGroupRequest(
        task_idx=0,
        n=n,
        client=EvalClientConfig(),
        model="model",
        sampling=SamplingConfig(),
    )
    return msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)
