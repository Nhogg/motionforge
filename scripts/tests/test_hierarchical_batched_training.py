"""Validate vectorized hierarchical rollout, GAE, and PPO flattening."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.league import FixedLeague, LeagueOpponent
from motionforge.models import (
    load_p2_checkpoint,
    migrate_p2_actor_parameters,
    p2_locomotion_normalization,
)
from motionforge.rollout import HierarchicalRolloutConfig, HierarchicalRolloutRunner
from motionforge.training import (
    HierarchicalPpoTrainer,
    HierarchicalPpoUpdater,
    HierarchicalTrainerConfig,
)


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_rebuilt_full_seed0/checkpoints/000200048640"
    )
    seed: int = 0
    num_environments: int = 4
    horizon: int = 10
    output: Path = Path("logs/hierarchical/batched_training_a.json")


def _tree_changed(left, right) -> bool:
    comparisons = jax.tree.map(
        lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)), left, right
    )
    return not all(jax.tree.leaves(comparisons))


def main(config: Config) -> None:
    if config.num_environments <= 1:
        raise ValueError("num_environments must exceed one")
    if config.num_environments % 2 != 0:
        raise ValueError("num_environments must be even")
    _, p2_parameters = load_p2_checkpoint(config.checkpoint)
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None)),
        config=MultiAgentTaskConfig(naconmax=256, njmax=512),
    )
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2_parameters[0]),
        HierarchicalRolloutConfig(
            learner_pursuer_probability=1.0,
            stochastic_actions=True,
        ),
    )
    initial = runner.reset(jax.random.PRNGKey(config.seed))
    parameters = migrate_p2_actor_parameters(
        runner.initialize_parameters(jax.random.PRNGKey(config.seed + 1), initial),
        p2_parameters[1],
    )
    environment.league = FixedLeague(
        LeagueOpponent("historical/batched-test", "historical", parameters)
    )
    updater = HierarchicalPpoUpdater()
    trainer = HierarchicalPpoTrainer(
        runner,
        updater,
        HierarchicalTrainerConfig(rollout_horizon=config.horizon),
    )
    train_state = updater.initialize(parameters)
    rollout_states = jax.vmap(runner.reset)(
        jax.random.split(jax.random.PRNGKey(config.seed + 2), config.num_environments),
        jp.arange(config.num_environments, dtype=jp.int32),
    )
    next_train_state, next_rollout_states, result = trainer.train_batch_iteration(
        train_state,
        rollout_states,
        seed=config.seed,
        episode_id=11,
    )
    next_rollout_states.environment.physics.data.qpos.block_until_ready()
    locomotion_samples = config.num_environments * config.horizon
    strategy_samples = locomotion_samples // 5
    environment_ids = np.asarray(result.batch.locomotion.environment_id)
    strategy_roles = np.asarray(result.batch.strategy.role)
    actions = np.asarray(result.batch.locomotion.action).reshape(
        config.num_environments, config.horizon, 29
    )
    checks = {
        "all_environments_advanced": bool(
            np.all(np.asarray(next_rollout_states.locomotion_steps) == config.horizon)
        ),
        "environment_ids_preserved": np.array_equal(
            np.unique(environment_ids), np.arange(config.num_environments)
        ),
        "independent_stochastic_rollouts": not np.array_equal(actions[0], actions[1]),
        "roles_balanced_in_single_update": (
            np.array_equal(np.unique(strategy_roles, axis=0), [[0.0, 1.0], [1.0, 0.0]])
            and np.allclose(strategy_roles.mean(axis=0), [0.5, 0.5])
            and result.trajectory.learner_role == "balanced"
        ),
        "locomotion_batch_flattened": (
            result.batch.locomotion.action.shape == (locomotion_samples, 29)
            and result.locomotion_targets.advantages.shape == (locomotion_samples,)
        ),
        "metrics_finite": all(
            bool(np.isfinite(np.asarray(value)).all())
            for value in jax.tree.leaves(result.update_metrics)
        ),
        "parameters_updated": (
            _tree_changed(
                train_state.parameters["strategy"],
                next_train_state.parameters["strategy"],
            )
            and _tree_changed(
                train_state.parameters["locomotion"],
                next_train_state.parameters["locomotion"],
            )
        ),
        "strategy_batch_flattened": (
            result.batch.strategy.command.shape == (strategy_samples, 3)
            and result.strategy_targets.advantages.shape == (strategy_samples,)
        ),
    }
    payload = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "output": str(config.output),
        },
        "experiment": "hierarchical_batched_training",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "samples": {
            "locomotion": locomotion_samples,
            "strategy": strategy_samples,
        },
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
