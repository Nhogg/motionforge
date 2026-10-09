"""Verify immutable, agent-specific policy routing for rollout segments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.league import FixedLeague, LeagueOpponent
from motionforge.models import LocomotionNormalization
from motionforge.rollout import HierarchicalRolloutConfig, HierarchicalRolloutRunner


@dataclass
class Config:
    seed: int = 0
    output: Path = Path("logs/hierarchical/opponent_routing_a.json")


def _tree_equal(left, right) -> bool:
    comparisons = jax.tree.map(
        lambda a, b: np.array_equal(np.asarray(a), np.asarray(b)), left, right
    )
    return all(jax.tree.leaves(comparisons))


def main(config: Config) -> None:
    environment = MultiAgentTaskEnv(
        league=FixedLeague(
            LeagueOpponent("placeholder", "scripted", None)
        ),
        config=MultiAgentTaskConfig(pursuer_index=1),
    )
    runner = HierarchicalRolloutRunner(
        environment,
        LocomotionNormalization(
            mean=jp.zeros(103, dtype=jp.float32),
            std=jp.ones(103, dtype=jp.float32),
        ),
        HierarchicalRolloutConfig(learner_pursuer_probability=1.0),
    )
    reset_key, learner_key, opponent_key = jax.random.split(
        jax.random.PRNGKey(config.seed), 3
    )
    state = runner.reset(reset_key)
    learner = runner.initialize_parameters(learner_key, state)
    opponent = runner.initialize_parameters(opponent_key, state)
    opponent_before = jax.tree.map(lambda value: np.asarray(value).copy(), opponent)
    environment.league = FixedLeague(
        LeagueOpponent("historical/0007", "historical", opponent)
    )
    segment = runner.start_segment(learner, seed=config.seed)
    next_state = jax.jit(runner.step)(segment, state)
    next_state.environment.physics.data.qpos.block_until_ready()

    checks = {
        "agent_actions_differ": not np.array_equal(
            np.asarray(next_state.previous_actions[0]),
            np.asarray(next_state.previous_actions[1]),
        ),
        "agent_commands_differ": not np.array_equal(
            np.asarray(next_state.commands[0]), np.asarray(next_state.commands[1])
        ),
        "learner_routed_to_agent1": segment.learner_index == 1,
        "learner_role_is_pursuer": segment.learner_role == "pursuer",
        "opponent_frozen": _tree_equal(opponent, opponent_before),
        "opponent_identity_fixed": (
            segment.opponent_id == "historical/0007"
            and segment.opponent_category == "historical"
        ),
        "opponent_routed_to_agent0": segment.opponent_index == 0,
        "opponent_role_is_evader": segment.opponent_role == "evader",
        "rollout_finite": bool(
            np.isfinite(np.asarray(next_state.environment.physics.data.qpos)).all()
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "experiment": "hierarchical_frozen_opponent_routing",
        "passed": all(checks.values()),
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
