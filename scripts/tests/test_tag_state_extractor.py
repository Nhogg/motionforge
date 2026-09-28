"""Audit the shared P7 TAG state-distribution extraction boundary on GPU."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.datasets import (
    TAG_STATE_DATASET_SCHEMA_VERSION,
    extract_tag_state_sample,
)
from motionforge.envs import TagEnvironmentConfig, TwoG1TagEnvironment


@dataclass
class Config:
    seed: int = 0
    output: Path = Path("logs/p7/tag_state_extractor_a.json")


def main(config: Config) -> None:
    environment = TwoG1TagEnvironment(TagEnvironmentConfig())
    state = jax.jit(environment.reset)(jax.random.PRNGKey(config.seed))
    commands = jp.asarray([[0.3, -0.2, 0.5], [-0.4, 0.1, -0.6]])
    actions = jp.reshape(jp.linspace(-1.0, 1.0, 58), (2, 29))
    targets = environment.default_joint_targets + 0.1 * actions
    extract = jax.jit(
        lambda current: extract_tag_state_sample(
            environment.model_bundle,
            current,
            commands,
            actions,
            targets,
        )
    )
    sample = extract(state)
    sample.root_position_world.block_until_ready()
    leaves = jax.tree.leaves(sample)

    checks = {
        "action_shape": sample.normalized_action.shape == (2, 29),
        "agent_distance_matches_reset": bool(
            np.isclose(np.asarray(sample.agent_distance), 2.0, atol=1e-5)
        ),
        "command_round_trip": bool(
            np.array_equal(
                np.asarray(sample.command_velocity), np.asarray(commands)
            )
        ),
        "finite_values": all(
            not np.issubdtype(np.asarray(value).dtype, np.number)
            or np.isfinite(np.asarray(value)).all()
            for value in leaves
        ),
        "gpu_backend": jax.default_backend() == "gpu",
        "initial_agents_not_fallen": not bool(np.asarray(sample.fallen).any()),
        "initial_agents_not_near_fall": not bool(np.asarray(sample.near_fall).any()),
        "joint_shapes": sample.joint_position.shape == (2, 29)
        and sample.joint_velocity.shape == (2, 29),
        "root_shapes": sample.root_position_world.shape == (2, 3)
        and sample.root_orientation_wxyz.shape == (2, 4),
        "schema_version": TAG_STATE_DATASET_SCHEMA_VERSION == 1,
        "target_shape": sample.joint_position_target.shape == (2, 29),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "p7_tag_state_extractor",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "schema_version": TAG_STATE_DATASET_SCHEMA_VERSION,
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
