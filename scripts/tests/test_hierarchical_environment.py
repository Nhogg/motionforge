"""Validate the WarpEnv -> LeagueEnv -> MultiAgentTaskEnv hierarchy."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import LeagueEnv, MultiAgentTaskConfig, MultiAgentTaskEnv, WarpEnv
from motionforge.league import FixedLeague, LeagueOpponent


@dataclass
class Config:
    seed: int = 0
    separation: float = 2.0
    naconmax: int = 64
    njmax: int = 256
    output: Path = Path("logs/hierarchical/environment_hierarchy_a.json")


def main(config: Config) -> None:
    opponent = LeagueOpponent(
        opponent_id="scripted/test",
        category="scripted",
        policy="test-policy-handle",
    )
    league = FixedLeague(opponent)
    environment = MultiAgentTaskEnv(
        league=league,
        config=MultiAgentTaskConfig(
            separation=config.separation,
            naconmax=config.naconmax,
            njmax=config.njmax,
        ),
    )
    selected = environment.sample_opponent(seed=config.seed)
    state = environment.reset(jax.random.PRNGKey(config.seed))
    next_state = jax.jit(environment.step)(
        state,
        environment.default_joint_targets,
    )

    qpos = np.asarray(next_state.physics.data.qpos)
    qvel = np.asarray(next_state.physics.data.qvel)
    checks = {
        "action_shape": environment.action_shape == (2, 29),
        "finite_state": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
        "inheritance": isinstance(environment, LeagueEnv)
        and isinstance(environment, WarpEnv),
        "league_identity": environment.league is league,
        "opponent_identity": selected is opponent,
        "physics_has_no_league_field": "league" not in WarpEnv.__dict__,
        "role_count": len(environment.roles.agents) == 2,
        "simulation_time": bool(
            np.isclose(
                float(np.asarray(next_state.physics.data.time)),
                environment.config.control_timestep,
            )
        ),
        "step_count": int(np.asarray(next_state.physics.step_count)) == 1,
        "task_observation_shape": (
            next_state.observation.relative_position.shape == (2, 2)
            and next_state.observation.relative_velocity.shape == (2, 2)
            and next_state.observation.arena_center_position.shape == (2, 2)
        ),
        "task_state_finite": bool(
            np.isfinite(np.asarray(next_state.observation.relative_position)).all()
            and np.isfinite(
                np.asarray(next_state.observation.relative_velocity)
            ).all()
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "environment_mro": [
            cls.__name__ for cls in type(environment).__mro__[:4]
        ],
        "experiment": "hierarchical_environment_inheritance",
        "jax_version": jax.__version__,
        "opponent": {
            "category": selected.category,
            "opponent_id": selected.opponent_id,
        },
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
