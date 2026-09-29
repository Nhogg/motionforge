"""Audit the accepted immutable learned evader used for distribution audits."""

from __future__ import annotations

import json
import platform
from dataclasses import FrozenInstanceError, asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import (
    TagEnvironmentConfig,
    TagPursuerConfig,
    TagPursuerEnvironment,
    pursuer_observation,
)
from motionforge.policies import FrozenLearnedEvader


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p7/training/tag_evader_fixed_noise_1m_seed0_retry/checkpoints/"
        "000000327680"
    )
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 0
    output: Path = Path("logs/p7/tag_frozen_evader_learned_a.json")


def main(config: Config) -> None:
    environment = TagPursuerEnvironment(
        locomotion_checkpoint=config.locomotion_checkpoint,
        evader_checkpoint=config.checkpoint,
        tag_config=TagEnvironmentConfig(pursuer_index=0),
        pursuer_config=TagPursuerConfig(),
    )
    opponent = FrozenLearnedEvader(
        agent_index=1,
        checkpoint_path=config.checkpoint,
    )
    same_opponent = FrozenLearnedEvader(
        agent_index=1,
        checkpoint_path=config.checkpoint,
    )
    changed_specification = FrozenLearnedEvader(
        agent_index=1,
        checkpoint_path=config.checkpoint,
        specification_version=2,
    )

    state = jax.jit(environment.reset)(jax.random.PRNGKey(config.seed))
    evader_observation = pursuer_observation(
        state.pipeline_state.tag_state.observation,
        1,
        np.zeros(3, dtype=np.float32),
        environment.tag_environment.config.arena_half_extent,
        environment.config.relative_velocity_scale,
    )
    command = jax.jit(opponent.command)(evader_observation)
    repeated_command = jax.jit(opponent.command)(evader_observation)
    command_host = np.asarray(command)
    next_state = jax.jit(environment.step)(state, np.zeros(3, dtype=np.float32))
    next_state.pipeline_state.tag_state.data.qpos.block_until_ready()

    immutable = False
    try:
        opponent.agent_index = 1
    except FrozenInstanceError:
        immutable = True

    checks = {
        "backend_gpu": jax.default_backend() == "gpu",
        "checkpoint_digest_recorded": len(
            opponent.specification()["checkpoint_digest"]
        )
        == 64,
        "command_deterministic": bool(
            np.array_equal(command_host, np.asarray(repeated_command))
        ),
        "command_finite": bool(np.isfinite(command_host).all()),
        "command_shape": command_host.shape == (3,),
        "command_within_limits": bool(
            np.all(np.abs(command_host) <= np.asarray([1.0, 0.5, 1.0]))
        ),
        "immutable": immutable,
        "environment_uses_learned_evader": isinstance(
            environment.evader, FrozenLearnedEvader
        ),
        "environment_resets_evader_command_memory": bool(
            np.array_equal(
                np.asarray(state.pipeline_state.evader_previous_command),
                np.zeros(3, dtype=np.float32),
            )
        ),
        "environment_tracks_evader_command": bool(
            np.array_equal(
                np.asarray(next_state.pipeline_state.evader_previous_command),
                command_host,
            )
        ),
        "same_checkpoint_same_fingerprint": (
            opponent.fingerprint == same_opponent.fingerprint
        ),
        "specification_changes_fingerprint": (
            opponent.fingerprint != changed_specification.fingerprint
        ),
        "specification_versioned": opponent.specification_version == 1,
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "command": command_host.tolist(),
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "locomotion_checkpoint": str(config.locomotion_checkpoint),
            "output": str(config.output),
        },
        "experiment": "p7_frozen_learned_evader",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "evader_fingerprint": opponent.fingerprint,
        "evader_specification": opponent.specification(),
        "python_version": platform.python_version(),
        "seed": config.seed,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
