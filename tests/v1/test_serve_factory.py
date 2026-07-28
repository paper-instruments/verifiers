import asyncio
import contextlib
import multiprocessing as mp
import queue

import pytest
from verifiers.v1.clients.config import EvalClientConfig
from verifiers.v1.env import EnvConfig, Environment
from verifiers.v1.serve import EnvClient, EnvServer, RunGroupRequest, serve_env
from verifiers.v1.types import SamplingConfig

FACTORY_CALLS = 0
NOT_CALLABLE = object()


class TrackingEnvironment(Environment):
    def __init__(self, config: EnvConfig) -> None:
        super().__init__(config)
        self.events: list[str] = []

    @contextlib.asynccontextmanager
    async def serving(self):
        self.events.append("serving")
        yield

    def episode(self, *args, **kwargs):
        self.events.append("episode")
        return TrackingEpisode()


class TrackingEpisode:
    async def run(self):
        return ["trace"]


def make_tracking_environment(label: str) -> Environment:
    global FACTORY_CALLS
    FACTORY_CALLS += 1
    assert label == "custom"
    return TrackingEnvironment(EnvConfig(taskset={"id": "echo-v1"}, harness={"id": "null"}))


def return_non_environment() -> str:
    return "not an environment"


def test_default_environment_construction_is_unchanged():
    server = EnvServer(
        config=EnvConfig(taskset={"id": "echo-v1"}, harness={"id": "null"}),
        address="tcp://127.0.0.1:0",
    )
    try:
        assert type(server.env) is Environment
        assert server.taskset_id == "echo-v1"
        assert server.num_tasks == 3
    finally:
        server.frontend.close()
        server.ctx.term()


def test_factory_constructs_custom_environment_once_and_preserves_lifecycle():
    global FACTORY_CALLS
    FACTORY_CALLS = 0
    server = EnvServer(
        factory_path=f"{__name__}.make_tracking_environment",
        factory_kwargs={"label": "custom"},
        address="tcp://127.0.0.1:0",
    )
    try:
        assert FACTORY_CALLS == 1
        assert isinstance(server.env, TrackingEnvironment)

        async def enter_serving() -> None:
            async with server.serving():
                response = await server._run_group(
                    RunGroupRequest(
                        task_idx=0,
                        n=2,
                        client=EvalClientConfig(),
                        model="model",
                        sampling=SamplingConfig(),
                    )
                )
                assert response.traces == ["trace"]

        server._context = lambda *args: None
        asyncio.run(enter_serving())
        assert server.env.events == ["serving", "episode"]
    finally:
        server.frontend.close()
        server.ctx.term()


@pytest.mark.parametrize(
    ("factory_path", "error"),
    [
        ("missing_factory_module.build", ModuleNotFoundError),
        (f"{__name__}.NOT_CALLABLE", TypeError),
        (f"{__name__}.return_non_environment", TypeError),
    ],
)
def test_invalid_factory_fails_server_construction(factory_path, error):
    with pytest.raises(error):
        EnvServer(factory_path=factory_path, address="tcp://127.0.0.1:0")


def test_config_and_factory_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        EnvServer(
            config=EnvConfig(taskset={"id": "echo-v1"}, harness={"id": "null"}),
            factory_path=f"{__name__}.make_tracking_environment",
            factory_kwargs={"label": "custom"},
            address="tcp://127.0.0.1:0",
        )


def test_factory_kwargs_without_factory_are_rejected():
    with pytest.raises(ValueError, match="factory_kwargs requires factory_path"):
        EnvServer(
            config=EnvConfig(taskset={"id": "echo-v1"}, harness={"id": "null"}),
            factory_kwargs={},
            address="tcp://127.0.0.1:0",
        )


def test_static_pool_preserves_factory_arguments(monkeypatch, tmp_path):
    module = tmp_path / "pool_factory.py"
    module.write_text(
        "import verifiers.v1 as vf\n"
        "\n"
        "def make_environment(taskset_id):\n"
        "    config = vf.EnvConfig(\n"
        "        taskset={'id': taskset_id},\n"
        "        harness={'id': 'null'},\n"
        "    )\n"
        "    return vf.Environment(config)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    process, address_queue = _start_server_process(
        max_workers=2,
        elastic=False,
        factory_path="pool_factory.make_environment",
        factory_kwargs={"taskset_id": "echo-v1"},
    )
    try:
        address = address_queue.get(timeout=30)

        async def fetch_info():
            client = EnvClient(address)
            try:
                return await client.info()
            finally:
                await client.close()

        info = asyncio.run(fetch_info())
        assert info.num_tasks == 3
        assert process.is_alive()
    finally:
        _stop_server_process(process, address_queue)


def test_pool_does_not_report_ready_when_factory_fails():
    process, address_queue = _start_server_process(
        max_workers=2,
        elastic=False,
        factory_path="builtins.dict",
        factory_kwargs={"value": "not an environment"},
    )
    try:
        with pytest.raises(queue.Empty):
            address_queue.get(timeout=5)
        process.join(timeout=10)
        assert not process.is_alive()
        assert process.exitcode != 0
    finally:
        _stop_server_process(process, address_queue)


def test_pool_does_not_report_ready_when_serving_fails(monkeypatch, tmp_path):
    module = tmp_path / "failing_serving_factory.py"
    module.write_text(
        "import contextlib\n"
        "import verifiers.v1 as vf\n"
        "\n"
        "class FailingServingEnvironment(vf.Environment):\n"
        "    @contextlib.asynccontextmanager\n"
        "    async def serving(self):\n"
        "        raise RuntimeError('serving failed')\n"
        "        yield\n"
        "\n"
        "def make_environment():\n"
        "    config = vf.EnvConfig(\n"
        "        taskset={'id': 'echo-v1'},\n"
        "        harness={'id': 'null'},\n"
        "    )\n"
        "    return FailingServingEnvironment(config)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    process, address_queue = _start_server_process(
        max_workers=2,
        elastic=False,
        factory_path="failing_serving_factory.make_environment",
    )
    try:
        with pytest.raises(queue.Empty):
            address_queue.get(timeout=5)
        process.join(timeout=10)
        assert not process.is_alive()
        assert process.exitcode != 0
    finally:
        _stop_server_process(process, address_queue)


def _start_server_process(**server_kwargs):
    context = mp.get_context("spawn")
    address_queue = context.Queue()
    process = context.Process(
        target=serve_env,
        kwargs={
            "address": "tcp://127.0.0.1:0",
            "address_queue": address_queue,
            **server_kwargs,
        },
        daemon=False,
    )
    process.start()
    return process, address_queue


def _stop_server_process(process, address_queue) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join()
    address_queue.close()
    address_queue.join_thread()
