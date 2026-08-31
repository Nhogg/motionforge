from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tyro
from dm_control.locomotion.examples import basic_cmu_2019


@dataclass(frozen=True)
class Config:
    """Reproduce the official dm_control CMU humanoid example."""

    seed: int = 0
    steps: int = 1000
    output: Path | None = None


def array_digest(array: np.ndarray) -> str:
    """Return a stable digest of a numeric array."""

    canonical = np.asarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def observation_shapes(
    observation: dict[str, Any],
) -> dict[str, list[int]]:
    return {
        name: list(np.asarray(value).shape)
        for name, value in sorted(observation.items())
    }


def run_reproduction(config: Config) -> dict[str, Any]:
    if config.steps <= 0:
        raise ValueError("--steps must be greater than zero")

    environment_random_state = np.random.RandomState(config.seed)
    action_rng = np.random.default_rng(config.seed)

    environment = basic_cmu_2019.cmu_humanoid_run_walls(
        random_state=environment_random_state
    )

    action_spec = environment.action_spec()
    timestep = environment.reset()

    initial_qpos = environment.physics.data.qpos.copy()
    initial_root_height = float(initial_qpos[2])
    initial_observation_shapes = observation_shapes(timestep.observation)

    total_reward = 0.0
    steps_executed = 0
    rewards_are_finite = True

    for _ in range(config.steps):
        action = action_rng.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape,
        )

        timestep = environment.step(action)
        steps_executed += 1

        reward = float(timestep.reward)
        total_reward += reward
        rewards_are_finite &= math.isfinite(reward)

        if timestep.last():
            break

    final_qpos = environment.physics.data.qpos.copy()
    final_qvel = environment.physics.data.qvel.copy()
    final_root_height = float(final_qpos[2])

    state_is_finite = bool(
        np.isfinite(final_qpos).all() and np.isfinite(final_qvel).all()
    )

    checks = {
        "action_shape": action_spec.shape == (56,),
        "environment_stepped": steps_executed > 0,
        "observations_present": len(timestep.observation) > 0,
        "rewards_finite": rewards_are_finite,
        "state_finite": state_is_finite,
    }

    result = {
        "action_shape": list(action_spec.shape),
        "checks": checks,
        "dm_control_version": version("dm-control"),
        "episode_ended": bool(timestep.last()),
        "experiment": "dm_control_cmu_humanoid_run_walls",
        "final_qpos_sha256": array_digest(final_qpos),
        "final_qvel_sha256": array_digest(final_qvel),
        "final_root_height": final_root_height,
        "initial_observation_shapes": initial_observation_shapes,
        "initial_qpos_sha256": array_digest(initial_qpos),
        "initial_root_height": initial_root_height,
        "mujoco_version": mujoco.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "requested_steps": config.steps,
        "seed": config.seed,
        "steps_executed": steps_executed,
        "total_reward": total_reward,
    }

    environment.close()
    return result


def main(config: Config) -> None:
    result = run_reproduction(config)
    serialized_result = json.dumps(result, sort_keys=True)

    print(serialized_result)

    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(serialized_result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(Config))
