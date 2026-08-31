from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tyro
from dm_control.locomotion import soccer


@dataclass(frozen=True)
class Config:
    """Reproduce the dm_control multi-agent soccer environment."""

    seed: int = 0
    steps: int = 100
    team_size: int = 2
    time_limit: float = 5.0
    walker_type: soccer.WalkerType = soccer.WalkerType.HUMANOID
    output: Path | None = None


def array_digest(array: np.ndarray) -> str:
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

    if config.team_size <= 0:
        raise ValueError("--team-size must be greater than zero")

    if config.time_limit <= 0.0:
        raise ValueError("--time-limit must be greater than zero")

    environment_random_state = np.random.RandomState(config.seed)
    action_rng = np.random.default_rng(config.seed)

    environment = soccer.load(
        team_size=config.team_size,
        time_limit=config.time_limit,
        random_state=environment_random_state,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=config.walker_type,
    )

    action_specs = environment.action_spec()
    timestep = environment.reset()

    expected_players = config.team_size * 2
    initial_qpos = environment.physics.data.qpos.copy()

    initial_observation_shapes = [
        observation_shapes(player_observation)
        for player_observation in timestep.observation
    ]

    total_rewards = np.zeros(expected_players, dtype=np.float64)
    rewards_are_finite = True
    steps_executed = 0

    for _ in range(config.steps):
        actions = [
            action_rng.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
                size=action_spec.shape,
            )
            for action_spec in action_specs
        ]

        timestep = environment.step(actions)
        steps_executed += 1

        rewards = np.asarray(timestep.reward, dtype=np.float64)
        total_rewards += rewards
        rewards_are_finite &= bool(np.isfinite(rewards).all())

        if timestep.last():
            break

    final_qpos = environment.physics.data.qpos.copy()
    final_qvel = environment.physics.data.qvel.copy()

    state_is_finite = bool(
        np.isfinite(final_qpos).all() and np.isfinite(final_qvel).all()
    )

    action_shapes = [list(action_spec.shape) for action_spec in action_specs]

    checks = {
        "action_spec_per_player": len(action_specs) == expected_players,
        "environment_stepped": steps_executed > 0,
        "observation_per_player": (len(timestep.observation) == expected_players),
        "rewards_finite": rewards_are_finite,
        "reward_per_player": len(timestep.reward) == expected_players,
        "state_finite": state_is_finite,
    }

    result = {
        "action_shapes": action_shapes,
        "checks": checks,
        "dm_control_version": version("dm-control"),
        "episode_ended": bool(timestep.last()),
        "experiment": "dm_control_multi_agent_soccer",
        "final_qpos_sha256": array_digest(final_qpos),
        "final_qvel_sha256": array_digest(final_qvel),
        "initial_observation_shapes": initial_observation_shapes,
        "initial_qpos_sha256": array_digest(initial_qpos),
        "mujoco_version": mujoco.__version__,
        "passed": all(checks.values()),
        "player_count": expected_players,
        "python_version": platform.python_version(),
        "requested_steps": config.steps,
        "seed": config.seed,
        "steps_executed": steps_executed,
        "team_size": config.team_size,
        "time_limit": config.time_limit,
        "total_rewards": total_rewards.tolist(),
        "walker_type": config.walker_type.name,
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
