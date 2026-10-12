"""Validate boundary-safe execution without corrupting PPO strategy actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskEnv
from motionforge.league import FixedLeague, LeagueOpponent
from motionforge.models import LocomotionNormalization
from motionforge.rollout import HierarchicalRolloutConfig, HierarchicalRolloutRunner


@dataclass
class Config:
    seed: int = 0
    output: Path = Path("logs/hierarchical/boundary_safety_a.json")


def main(config: Config) -> None:
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None))
    )
    runner = HierarchicalRolloutRunner(
        environment,
        LocomotionNormalization(
            mean=jp.zeros(103, dtype=jp.float32),
            std=jp.ones(103, dtype=jp.float32),
        ),
        HierarchicalRolloutConfig(
            boundary_command_safety_enabled=True,
            boundary_command_margin=1.0,
        ),
    )
    state = runner.reset(jax.random.PRNGKey(config.seed))
    positions = jp.asarray([[4.0, 0.0], [-3.5, 0.0]], dtype=jp.float32)
    qpos = state.environment.physics.data.qpos
    for agent in runner.locomotion_layout.agents:
        qpos = qpos.at[agent.qpos_slice.start + 3 : agent.qpos_slice.start + 7].set(
            jp.asarray([1.0, 0.0, 0.0, 0.0])
        )
    state = state.replace(
        environment=state.environment.replace(
            physics=state.environment.physics.replace(
                data=state.environment.physics.data.replace(qpos=qpos)
            ),
            diagnostics=state.environment.diagnostics.replace(
                planar_position=positions
            ),
        )
    )
    strategy_actions = jp.asarray(
        [[1.0, 0.2, 0.3], [-1.0, 0.2, -0.3]], dtype=jp.float32
    )
    commands = runner._project_boundary_safe_commands(state, strategy_actions)
    commands = np.asarray(commands)
    raw = np.asarray(strategy_actions)

    checks = {
        "finite": bool(np.isfinite(commands).all()),
        "outward_removed_at_edge": bool(np.isclose(commands[0, 0], 0.0)),
        "outward_attenuated_in_margin": bool(np.isclose(commands[1, 0], -0.5)),
        "tangential_and_yaw_preserved": bool(np.allclose(commands[:, 1:], raw[:, 1:])),
        "raw_strategy_actions_preserved": bool(
            np.array_equal(np.asarray(strategy_actions), raw)
        ),
    }
    result = {
        "checks": checks,
        "commands": commands.tolist(),
        "experiment": "hierarchical_boundary_command_safety",
        "passed": all(checks.values()),
        "strategy_actions": raw.tolist(),
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
