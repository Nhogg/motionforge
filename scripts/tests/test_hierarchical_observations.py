"""Validate task and P2 observation adapters for both hierarchy levels."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.controllers import (
    build_g1_tag_policy_observation_layout,
    g1_tag_policy_observation,
)
from motionforge.envs import (
    HierarchicalObservationConfig,
    MultiAgentTaskConfig,
    MultiAgentTaskEnv,
    locomotion_observations,
    role_encodings,
    strategy_observations,
)
from motionforge.league import FixedLeague, LeagueOpponent


@dataclass
class Config:
    seed: int = 0
    pursuer_index: int = 0
    relative_velocity_scale: float = 2.0
    output: Path = Path("logs/hierarchical/observations_a.json")


def main(config: Config) -> None:
    league = FixedLeague(
        LeagueOpponent(
            opponent_id="scripted/test",
            category="scripted",
            policy="test-policy-handle",
        )
    )
    environment = MultiAgentTaskEnv(
        league=league,
        config=MultiAgentTaskConfig(pursuer_index=config.pursuer_index),
    )
    state = environment.reset(jax.random.PRNGKey(config.seed))
    layout = build_g1_tag_policy_observation_layout(environment.model_bundle)
    previous_commands = jp.asarray(
        [[0.25, -0.10, 0.50], [-0.40, 0.20, -0.25]], dtype=jp.float32
    )
    previous_actions = jp.linspace(-0.5, 0.5, 58).reshape(2, 29)
    phases = jp.asarray([[0.25, 0.75], [1.25, 1.75]], dtype=jp.float32)
    roles = role_encodings(jp.asarray(config.pursuer_index))
    observation_config = HierarchicalObservationConfig(
        arena_half_extent=environment.config.arena_half_extent,
        relative_velocity_scale=config.relative_velocity_scale,
    )

    build_strategy = jax.jit(
        lambda current_state: strategy_observations(
            current_state,
            previous_commands,
            roles,
            observation_config,
        )
    )
    build_locomotion = jax.jit(
        lambda current_state: locomotion_observations(
            current_state,
            layout,
            previous_actions,
            phases,
            environment.default_joint_targets,
        )
    )
    strategy = build_strategy(state)
    locomotion = build_locomotion(state)
    strategy_array = strategy.as_array()
    locomotion_array = locomotion.as_array(previous_commands)

    legacy_locomotion = jp.stack(
        [
            g1_tag_policy_observation(
                state.physics.data,
                layout,
                agent_index,
                previous_commands[agent_index],
                previous_actions[agent_index],
                phases[agent_index],
                environment.default_joint_targets[agent_index],
            )["state"]
            for agent_index in range(2)
        ]
    )
    expected_strategy_without_role = jp.concatenate(
        (
            state.observation.relative_position
            / (2.0 * environment.config.arena_half_extent),
            state.observation.relative_velocity / config.relative_velocity_scale,
            state.observation.arena_center_position
            / environment.config.arena_half_extent,
            previous_commands / jp.asarray([1.0, 0.5, 1.0]),
        ),
        axis=-1,
    )

    strategy_host = np.asarray(strategy_array)
    locomotion_host = np.asarray(locomotion_array)
    legacy_host = np.asarray(legacy_locomotion)
    checks = {
        "locomotion_finite": bool(np.isfinite(locomotion_host).all()),
        "locomotion_matches_p2_adapter": bool(
            np.allclose(locomotion_host, legacy_host, rtol=0.0, atol=1e-7)
        ),
        "locomotion_shape": locomotion_host.shape == (2, 103),
        "no_task_fields_in_locomotion": locomotion_array.shape[-1] == 103,
        "role_complement": bool(
            np.array_equal(np.asarray(roles.sum(axis=0)), np.asarray([1.0, 1.0]))
        ),
        "role_encoding": bool(
            np.array_equal(
                np.asarray(roles[config.pursuer_index]),
                np.asarray([1.0, 0.0]),
            )
            and np.array_equal(
                np.asarray(roles[1 - config.pursuer_index]),
                np.asarray([0.0, 1.0]),
            )
        ),
        "strategy_finite": bool(np.isfinite(strategy_host).all()),
        "strategy_matches_legacy_scaling": bool(
            np.array_equal(
                strategy_host[:, :9],
                np.asarray(expected_strategy_without_role),
            )
        ),
        "strategy_shape": strategy_host.shape == (2, 11),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "hierarchical_observation_adapters",
        "jax_version": jax.__version__,
        "maximum_locomotion_parity_error": float(
            np.max(np.abs(locomotion_host - legacy_host))
        ),
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "role_encodings": np.asarray(roles).tolist(),
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
