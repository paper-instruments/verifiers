import asyncio
from types import SimpleNamespace

import pytest

import verifiers.v1.agent as agent_module
from verifiers.v1.agent import Agent
from verifiers.v1.rollout import RolloutRun


class CleanupEscape(BaseException):
    pass


class RecordingStack:
    def __init__(self, events: list, error: BaseException | None = None) -> None:
        self.events = events
        self.error = error

    async def aclose(self) -> None:
        self.events.append("stack")
        if self.error is not None:
            raise self.error


class RecordingRuntime:
    def __init__(self, events: list, error: BaseException | None = None) -> None:
        self.events = events
        self.error = error

    async def stop(self) -> None:
        self.events.append("runtime")
        if self.error is not None:
            raise self.error


class RecordingHarness:
    def __init__(self, events: list, *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    async def abort(self, trace, error: BaseException) -> None:
        self.events.append(("harness", trace.id, error))
        if self.fail:
            raise RuntimeError("observer failed")


async def test_rollout_abort_notifies_harness_after_teardown_once():
    events = []
    error = asyncio.CancelledError()
    run = _run(events)

    await run.abort(error)
    await run.abort(error)

    assert events == [
        "stack",
        "runtime",
        ("harness", "trace", error),
    ]


async def test_rollout_abort_hook_failure_does_not_replace_cancellation():
    events = []
    error = asyncio.CancelledError()
    run = _run(events, fail=True)

    await run.abort(error)

    assert events[-1] == ("harness", "trace", error)


@pytest.mark.parametrize("location", ["stack", "runtime"])
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, CleanupEscape])
async def test_rollout_abort_cleanup_escape_preserves_original(location, error_type):
    events = []
    original = asyncio.CancelledError("original")
    cleanup_error = error_type("cleanup")
    run = _run(
        events,
        stack_error=cleanup_error if location == "stack" else None,
        runtime_error=cleanup_error if location == "runtime" else None,
    )

    async def escape():
        try:
            raise original
        except BaseException as error:
            await run.abort(error)
            raise

    with pytest.raises(asyncio.CancelledError) as raised:
        await escape()

    assert raised.value is original
    assert events[-1] == ("harness", "trace", original)


async def test_repeated_cancellation_waits_for_cleanup_and_preserves_original(caplog):
    events = []
    original = asyncio.CancelledError("original")
    started = asyncio.Event()
    allowed = asyncio.Event()
    run = _run(events)

    async def blocking_close() -> None:
        events.append("stack")
        started.set()
        await allowed.wait()

    run._stack.aclose = blocking_close

    async def escape():
        try:
            raise original
        except BaseException as error:
            await run.abort(error)
            raise

    task = asyncio.create_task(escape())
    await asyncio.wait_for(started.wait(), 1)
    task.cancel("repeated")
    await asyncio.sleep(0)
    assert not task.done()

    allowed.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.wait_for(task, 1)

    assert raised.value is original
    assert events[-1] == ("harness", "trace", original)
    assert "rollout abort cleanup failed" not in caplog.text


async def test_agent_forwards_base_exception_to_rollout_abort(monkeypatch):
    error = asyncio.CancelledError()
    observed = []

    class EscapingRun:
        def __init__(self, **_kwargs) -> None:
            return None

        async def open(self) -> bool:
            raise error

        async def abort(self, escaped: BaseException) -> None:
            observed.append(escaped)

    monkeypatch.setattr(agent_module, "RolloutRun", EscapingRun)
    agent = object.__new__(Agent)
    agent._rollout_params = lambda *_args: {}

    with pytest.raises(asyncio.CancelledError) as raised:
        await agent._run_once(SimpleNamespace(), None, None, None)

    assert raised.value is error
    assert observed == [error]


async def test_agent_does_not_abort_a_controlled_rollout_stop(monkeypatch):
    observed = []
    trace = SimpleNamespace(runtime=None)

    class StoppedRun:
        def __init__(self, **_kwargs) -> None:
            return None

        async def open(self) -> bool:
            return True

        async def step(self) -> bool:
            return False

        async def close(self):
            return trace

        async def abort(self, _error: BaseException) -> None:
            observed.append("abort")

    monkeypatch.setattr(agent_module, "RolloutRun", StoppedRun)
    agent = object.__new__(Agent)
    agent._rollout_params = lambda *_args: {}

    result = await agent._run_once(SimpleNamespace(), None, None, None)

    assert result is trace
    assert observed == []


def _run(
    events: list,
    *,
    fail: bool = False,
    stack_error: BaseException | None = None,
    runtime_error: BaseException | None = None,
) -> RolloutRun:
    run = object.__new__(RolloutRun)
    run._aborted = False
    run._closed = False
    run._stack = RecordingStack(events, stack_error)
    run._owns_runtime = True
    run.runtime = RecordingRuntime(events, runtime_error)
    run.harness = RecordingHarness(events, fail=fail)
    run.trace = SimpleNamespace(id="trace")
    return run
