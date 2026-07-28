import asyncio
import contextlib

import msgpack
import pytest
import zmq
from verifiers.v1.clients.config import EvalClientConfig
from verifiers.v1.env import EnvConfig
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


class BlockingEpisode:
    def __init__(self, state: CancellationState) -> None:
        self.state = state

    async def run(self):
        self.state.started.set()
        try:
            await asyncio.Future()
        finally:
            self.state.cleanup_started.set()
            await self.state.cleanup_allowed.wait()
            self.state.cleaned.set()


@pytest.mark.parametrize("method", ["run_rollout", "run_group"])
async def test_client_cancellation_waits_for_remote_episode_cleanup(method):
    state = CancellationState()
    server = _blocking_server(state)
    server_task = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    request_task = asyncio.create_task(_run(client, method))
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


async def test_server_shutdown_waits_for_request_cleanup():
    state = CancellationState()
    server = _blocking_server(state)
    server_task = asyncio.create_task(server.run())
    ctx = zmq.asyncio.Context()
    socket = ctx.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(server.address)
    request = RunGroupRequest(
        task_idx=0,
        n=2,
        client=EvalClientConfig(),
        model="model",
        sampling=SamplingConfig(),
    )
    try:
        await socket.send_multipart(
            [
                b"shutdown-target",
                RunGroupRequest.method.encode(),
                msgpack.packb(request.model_dump(mode="json"), use_bin_type=True),
            ]
        )
        await asyncio.wait_for(state.started.wait(), 2)
        server_task.cancel()
        await asyncio.wait_for(state.cleanup_started.wait(), 2)
        server_task.cancel()
        await asyncio.sleep(0)
        assert not server_task.done()

        state.cleanup_allowed.set()
        await asyncio.wait_for(server_task, 2)
        assert state.cleaned.is_set()
    finally:
        state.cleanup_allowed.set()
        if not server_task.done():
            server_task.cancel()
            await asyncio.wait_for(server_task, 2)
        socket.close()
        ctx.term()


async def test_cancel_unknown_target_is_a_successful_noop():
    state = CancellationState()
    server = _blocking_server(state)
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


async def test_real_pool_routes_cancel_to_owner_and_releases_capacity(monkeypatch, tmp_path):
    _write_pool_factory(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    pool = EnvServerPool(
        server_kwargs={
            "factory_path": "pool_cancel_factory.make_environment",
            "factory_kwargs": {"state_dir": str(tmp_path)},
        },
        max_workers=2,
        address="tcp://127.0.0.1:0",
        legacy=False,
        elastic=False,
    )
    pool.start()
    pool_task = asyncio.create_task(pool.run())
    clients = [EnvClient(pool.address), EnvClient(pool.address)]
    requests = []
    try:
        unknown = await asyncio.wait_for(clients[0].cancel("missing"), 2)
        assert not unknown.cancelled
        assert pool.pending == {}

        requests = [asyncio.create_task(_run(client, "run_group")) for client in clients]
        await _wait_for_files(tmp_path, "started-*", count=2)
        requests[0].cancel()
        await _wait_for_files(tmp_path, "cleanup-started-*", count=1)
        assert not requests[0].done()
        assert not requests[1].done()

        (tmp_path / "allow-cleanup").touch()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(requests[0], 2)
        assert len(list(tmp_path.glob("cleaned-*"))) == 1
        assert len(pool.pending) == 1
        assert pool.in_flight == 2
        assert sum(worker["active"] for worker in pool.workers) == 2
        assert not requests[1].done()

        requests[1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(requests[1], 2)
        await _wait_for_files(tmp_path, "cleaned-*", count=2)
        assert pool.pending == {}
        assert pool.in_flight == 0
        assert sum(worker["active"] for worker in pool.workers) == 0
    finally:
        (tmp_path / "allow-cleanup").touch()
        for request in requests:
            if not request.done():
                request.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        for client in clients:
            await client.close()
        pool_task.cancel()
        await asyncio.wait_for(pool_task, 10)


async def test_real_pool_natural_completion_race_releases_capacity(monkeypatch, tmp_path):
    _write_pool_factory(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "natural-mode").touch()
    pool = EnvServerPool(
        server_kwargs={
            "factory_path": "pool_cancel_factory.make_environment",
            "factory_kwargs": {"state_dir": str(tmp_path)},
        },
        max_workers=2,
        address="tcp://127.0.0.1:0",
        legacy=False,
        elastic=False,
    )
    pool.start()
    pool_task = asyncio.create_task(pool.run())
    ctx = zmq.asyncio.Context()
    socket = ctx.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(pool.address)
    target_id = b"target"
    cancel_id = b"cancel"
    try:
        await socket.send_multipart([target_id, b"run_group", _run_group_payload(n=3)])
        await _wait_for_files(tmp_path, "started-*", count=1)
        (tmp_path / "allow-natural-completion").touch()
        await _wait_for_files(tmp_path, "cleaned-*", count=1)
        await socket.send_multipart([cancel_id, b"cancel", _cancel_payload(target_id)])

        for _ in range(2):
            request_id, data = await asyncio.wait_for(socket.recv_multipart(), 2)
            if request_id == cancel_id:
                response = CancelResponse.model_validate(msgpack.unpackb(data, raw=False))
                assert response.success
                break
        else:
            pytest.fail("pool did not acknowledge cancellation")

        assert pool.pending == {}
        assert pool.in_flight == 0
        assert sum(worker["active"] for worker in pool.workers) == 0
    finally:
        socket.close()
        ctx.term()
        pool_task.cancel()
        await asyncio.wait_for(pool_task, 10)


async def _run(client: EnvClient, method: str):
    kwargs = {
        "task_idx": 0,
        "client": EvalClientConfig(),
        "model": "model",
        "sampling": SamplingConfig(),
    }
    if method == "run_group":
        return await client.run_group(n=2, **kwargs)
    return await client.run_rollout(**kwargs)


def _blocking_server(state: CancellationState) -> EnvServer:
    server = EnvServer(
        config=EnvConfig(taskset={"id": "echo-v1"}, harness={"id": "null"}),
        address="tcp://127.0.0.1:0",
    )
    server._context = lambda *args: None
    server.env.episode = lambda *args, **kwargs: BlockingEpisode(state)

    @contextlib.asynccontextmanager
    async def serving():
        yield

    server.env.serving = serving
    return server


def _run_group_payload(n: int) -> bytes:
    request = RunGroupRequest(
        task_idx=0,
        n=n,
        client=EvalClientConfig(),
        model="model",
        sampling=SamplingConfig(),
    )
    return msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)


def _cancel_payload(target_request_id: bytes) -> bytes:
    request = CancelRequest(target_request_id=target_request_id.decode())
    return msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)


async def _wait_for_files(path, pattern: str, count: int) -> None:
    async with asyncio.timeout(10):
        while len(list(path.glob(pattern))) < count:
            await asyncio.sleep(0.01)


def _write_pool_factory(path) -> None:
    (path / "pool_cancel_factory.py").write_text(
        """\
import asyncio
import contextlib
import os
from pathlib import Path

import verifiers.v1 as vf


class BlockingEpisode:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)

    async def run(self):
        pid = os.getpid()
        (self.state_dir / f"started-{pid}").touch()
        natural = (self.state_dir / "natural-mode").exists()
        try:
            if natural:
                while not (
                    self.state_dir / "allow-natural-completion"
                ).exists():
                    await asyncio.sleep(0.01)
                return []
            await asyncio.Future()
        finally:
            (self.state_dir / f"cleanup-started-{pid}").touch()
            if not natural:
                while not (self.state_dir / "allow-cleanup").exists():
                    await asyncio.sleep(0.01)
            (self.state_dir / f"cleaned-{pid}").touch()


class BlockingEnvironment(vf.Environment):
    def __init__(self, state_dir):
        config = vf.EnvConfig(
            taskset={"id": "echo-v1"},
            harness={"id": "null"},
        )
        super().__init__(config)
        self.state_dir = state_dir

    @contextlib.asynccontextmanager
    async def serving(self):
        yield

    def episode(self, *args, **kwargs):
        return BlockingEpisode(self.state_dir)


def make_environment(state_dir):
    return BlockingEnvironment(state_dir)
"""
    )
