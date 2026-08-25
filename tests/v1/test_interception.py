import asyncio

import pytest
from aiohttp import web

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession


class RecordingClient(vf.Client):
    def __init__(self) -> None:
        self.closed = False

    async def get_response(self, *args, **kwargs):
        raise AssertionError("test client received an unexpected model request")

    async def close(self) -> None:
        self.closed = True


async def test_registered_client_factory_is_serializable_and_server_owned():
    clients: list[RecordingClient] = []

    def factory() -> RecordingClient:
        client = RecordingClient()
        clients.append(client)
        return client

    with vf.register_client_factory(factory) as config:
        assert vf.AgentConfig(client=config).model_dump(mode="json")["client"] == {
            "type": "registered",
            "key": config.key,
        }
        sessions = [
            RolloutSession(
                ModelContext(model="model", client=config),
                vf.Trace(
                    agent=vf.AgentInfo(config=vf.AgentConfig()),
                    task=vf.TraceTask(
                        type="Task", data=vf.TaskData(idx=i, prompt="test")
                    ),
                ),
            )
            for i in range(2)
        ]

        async with (
            InterceptionServer() as server,
            server.acquire(sessions[0]),
            server.acquire(sessions[1]),
        ):
            assert sessions[0].client is not sessions[1].client

        assert len(clients) == 2
        assert all(client.closed for client in clients)

    with pytest.raises(RuntimeError, match="No client factory is registered"):
        vf.resolve_client(config)


async def test_cancelled_rollout_releases_its_router_session(unused_tcp_port):
    released = []

    async def release(request: web.Request) -> web.Response:
        released.append(request.headers["X-Session-ID"])
        return web.Response(status=204)

    app = web.Application()
    app.router.add_delete("/v1/router/session", release)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()

    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="test")),
    )
    session = RolloutSession(
        ModelContext(
            model="model",
            client=EvalClientConfig(
                base_url=f"http://127.0.0.1:{unused_tcp_port}/v1",
                session_release_path="/v1/router/session",
            ),
        ),
        trace,
    )

    try:
        with pytest.raises(asyncio.CancelledError):
            async with InterceptionServer() as server, server.acquire(session):
                raise asyncio.CancelledError
    finally:
        await runner.cleanup()

    assert session.released
    assert released == [trace.id]


async def test_router_release_failure_is_retried_without_failing_cleanup(
    unused_tcp_port,
):
    attempts = 0

    async def release(_request: web.Request) -> web.Response:
        nonlocal attempts
        attempts += 1
        return web.Response(status=503)

    app = web.Application()
    app.router.add_delete("/v1/router/session", release)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()

    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="test")),
    )
    session = RolloutSession(
        ModelContext(
            model="model",
            client=EvalClientConfig(
                base_url=f"http://127.0.0.1:{unused_tcp_port}/v1",
                session_release_path="/v1/router/session",
            ),
        ),
        trace,
    )

    try:
        async with InterceptionServer() as server, server.acquire(session):
            pass
    finally:
        await runner.cleanup()

    assert session.released
    assert attempts == 3
