"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig);
`endpoints.toml` isn't an eval config, and `configs/eval|rl|gepa/` are the legacy
`vf-eval` / training formats (different, non-v1 config classes), so both are out of scope here.
"""

import tomllib
from pathlib import Path

import pytest

from verifiers.v1.clients.config import EvalClientConfig
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.serve.types import RunRequest
from verifiers.v1.types import SamplingConfig

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    # resolved to a v1 taskset or a v0 env id
    assert config.env.taskset.id or config.id


def test_run_request_carries_initial_trace_info() -> None:
    request = RunRequest(
        task_data={"idx": 3, "prompt": "hello"},
        trace_info={"rollout_group_id": "group-7", "sampled_policy_version": 4},
        client=EvalClientConfig(),
        model="model",
        sampling=SamplingConfig(),
    )

    restored = RunRequest.model_validate(request.model_dump(mode="json"))

    assert restored.trace_info == {
        "rollout_group_id": "group-7",
        "sampled_policy_version": 4,
    }
