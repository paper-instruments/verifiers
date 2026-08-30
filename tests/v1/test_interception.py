import asyncio

import pytest
from aiohttp import web

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession


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
