"""Audit the accepted immutable learned pursuer used for evader training."""

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
)
from motionforge.policies import FrozenLearnedPursuer


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p7/training/tag_pursuer_fixed_noise_1m_seed0/checkpoints/"
        "000001146880"
    )
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 0
    output: Path = Path("logs/p7/tag_frozen_pursuer_a.json")


def main(config: Config) -> None:
    environment = TagPursuerEnvironment(
        locomotion_checkpoint=config.locomotion_checkpoint,
        tag_config=TagEnvironmentConfig(pursuer_index=0),
        pursuer_config=TagPursuerConfig(),
    )
    opponent = FrozenLearnedPursuer(
        agent_index=environment.pursuer_index,
        checkpoint_path=config.checkpoint,
    )
    same_opponent = FrozenLearnedPursuer(
        agent_index=environment.pursuer_index,
        checkpoint_path=config.checkpoint,
    )
    changed_specification = FrozenLearnedPursuer(
        agent_index=environment.pursuer_index,
        checkpoint_path=config.checkpoint,
        specification_version=2,
    )

    state = jax.jit(environment.reset)(jax.random.PRNGKey(config.seed))
    command = jax.jit(opponent.command)(state.obs)
    repeated_command = jax.jit(opponent.command)(state.obs)
    command_host = np.asarray(command)

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
        "experiment": "p7_frozen_learned_pursuer",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "pursuer_fingerprint": opponent.fingerprint,
        "pursuer_specification": opponent.specification(),
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
