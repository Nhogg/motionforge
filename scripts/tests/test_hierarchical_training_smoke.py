"""Run one complete hierarchical PPO iteration and checkpoint its result."""

from __future__ import annotations

import json
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskEnv
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
    HierarchicalUpdateConfig,
    load_hierarchical_checkpoint,
    save_hierarchical_checkpoint,
)


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    horizon: int = 10
    output: Path = Path("logs/hierarchical/training_smoke_a.json")


def _tree_equal(left, right) -> bool:
    comparisons = jax.tree.map(
        lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)), left, right
    )
    return all(jax.tree.leaves(comparisons))


def _tree_changed(left, right) -> bool:
    return not _tree_equal(left, right)


def main(config: Config) -> None:
    _, p2_parameters = load_p2_checkpoint(config.checkpoint)
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None))
    )
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2_parameters[0]),
        HierarchicalRolloutConfig(
            learner_pursuer_probability=1.0,
            stochastic_actions=True,
        ),
    )
    reset_key, parameter_key = jax.random.split(jax.random.PRNGKey(config.seed))
    rollout_state = runner.reset(reset_key)
    parameters = migrate_p2_actor_parameters(
        runner.initialize_parameters(parameter_key, rollout_state),
        p2_parameters[1],
    )
    frozen_opponent = jax.tree.map(lambda value: value, parameters)
    environment.league = FixedLeague(
        LeagueOpponent("historical/training-smoke", "historical", frozen_opponent)
    )
    update_config = HierarchicalUpdateConfig()
    updater = HierarchicalPpoUpdater(update_config)
    trainer = HierarchicalPpoTrainer(
        runner,
        updater,
        HierarchicalTrainerConfig(rollout_horizon=config.horizon),
    )
    initial_train_state = updater.initialize(parameters)
    train_state, final_rollout_state, result = trainer.train_iteration(
        initial_train_state,
        rollout_state,
        seed=config.seed,
        episode_id=0,
        environment_id=0,
    )
    final_rollout_state.environment.physics.data.qpos.block_until_ready()

    with tempfile.TemporaryDirectory(prefix="motionforge-training-smoke-") as root:
        checkpoint = save_hierarchical_checkpoint(
            Path(root) / "000000000001",
            train_state,
            update_config=update_config,
            metadata={"seed": config.seed, "horizon": config.horizon},
        )
        restored, manifest = load_hierarchical_checkpoint(
            checkpoint, initial_train_state
        )

    metric_values = jax.tree.leaves(result.update_metrics)
    checks = {
        "checkpoint_round_trip": _tree_equal(train_state, restored),
        "checkpoint_step_recorded": manifest["iteration"] == 1,
        "frozen_opponent_unchanged": _tree_equal(frozen_opponent, parameters),
        "learner_parameters_updated": (
            _tree_changed(
                initial_train_state.parameters["strategy"],
                train_state.parameters["strategy"],
            )
            and _tree_changed(
                initial_train_state.parameters["locomotion"],
                train_state.parameters["locomotion"],
            )
        ),
        "metrics_finite": all(
            bool(np.isfinite(np.asarray(value)).all()) for value in metric_values
        ),
        "optimizer_counters": (
            int(train_state.iteration) == 1
            and int(train_state.strategy_updates) == 1
            and int(train_state.locomotion_updates) == 1
        ),
        "rollout_completed": (
            int(final_rollout_state.locomotion_steps) == config.horizon
            and result.batch.locomotion.action.shape == (config.horizon, 29)
            and result.batch.strategy.command.shape == (config.horizon // 5, 3)
        ),
        "rollout_finite": all(
            bool(np.isfinite(np.asarray(value)).all())
            for value in jax.tree.leaves(result.batch)
        ),
        "stochastic_log_probabilities": bool(
            np.any(np.asarray(result.batch.strategy.log_probability) != 0.0)
            and np.any(np.asarray(result.batch.locomotion.log_probability) != 0.0)
        ),
        "trajectory_provenance": (
            result.trajectory.opponent_id == "historical/training-smoke"
            and result.trajectory.opponent_category == "historical"
            and result.trajectory.learner_role == "pursuer"
        ),
    }
    result_payload = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "output": str(config.output),
        },
        "experiment": "hierarchical_end_to_end_training_smoke",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "samples": {
            "locomotion": config.horizon,
            "strategy": config.horizon // 5,
        },
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result_payload, sort_keys=True))
    if not result_payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
