"""Validate the immutable scripted opponent used for P7 pursuer training."""

from __future__ import annotations

import json
import platform
from dataclasses import FrozenInstanceError, asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import TagEnvironmentConfig, TagRole, TwoG1TagEnvironment
from motionforge.policies import FrozenScriptedEvader, ScriptedEvaderConfig


@dataclass
class Config:
    seed: int = 0
    pursuer_index: int = 0
    output: Path = Path("logs/p7/tag_frozen_evader_a.json")


def main(config: Config) -> None:
    environment = TwoG1TagEnvironment(
        TagEnvironmentConfig(pursuer_index=config.pursuer_index)
    )
    evader_index = environment.roles.by_role(TagRole.EVADER).index
    opponent = FrozenScriptedEvader(agent_index=evader_index)
    same_opponent = FrozenScriptedEvader(agent_index=evader_index)
    changed_opponent = FrozenScriptedEvader(
        agent_index=evader_index,
        config=ScriptedEvaderConfig(linear_gain=0.7),
    )

    state = jax.jit(environment.reset)(jax.random.PRNGKey(config.seed))
    command = jax.jit(opponent.command)(state.observation)
    repeated_command = jax.jit(opponent.command)(state.observation)
    command_host = np.asarray(command)
    repeated_host = np.asarray(repeated_command)

    immutable = False
    try:
        opponent.agent_index = config.pursuer_index
    except FrozenInstanceError:
        immutable = True

    policy_config = opponent.config
    checks = {
        "backend_gpu": jax.default_backend() == "gpu",
        "command_deterministic": bool(np.array_equal(command_host, repeated_host)),
        "command_finite": bool(np.isfinite(command_host).all()),
        "command_shape": command_host.shape == (3,),
        "command_within_limits": bool(
            abs(command_host[0]) <= policy_config.maximum_forward_speed
            and abs(command_host[1]) <= policy_config.maximum_lateral_speed
            and abs(command_host[2]) <= policy_config.maximum_yaw_rate
        ),
        "configuration_changes_fingerprint": (
            opponent.fingerprint != changed_opponent.fingerprint
        ),
        "immutable": immutable,
        "role_binding": evader_index != config.pursuer_index,
        "same_configuration_same_fingerprint": (
            opponent.fingerprint == same_opponent.fingerprint
        ),
        "specification_versioned": opponent.specification_version == 1,
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "command": command_host.tolist(),
        "config": {**asdict(config), "output": str(config.output)},
        "evader_fingerprint": opponent.fingerprint,
        "evader_specification": opponent.specification(),
        "experiment": "p7_frozen_scripted_evader",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "seed": config.seed,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
