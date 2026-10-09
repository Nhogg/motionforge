"""Validate separate optimizer state and hierarchical update cadence."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.models import (
    HierarchicalPolicy,
    LocomotionObservation,
    StrategyObservation,
    tanh_normal_log_prob,
)
from motionforge.rollout import AdvantageTargets, LocomotionBatch, StrategyBatch
from motionforge.training import HierarchicalPpoUpdater, HierarchicalUpdateConfig


@dataclass
class Config:
    seed: int = 0
    batch_size: int = 8
    strategy_update_interval: int = 2
    locomotion_update_interval: int = 1
    output: Path = Path("logs/hierarchical/update_schedule_a.json")


def _observations(batch_size: int):
    strategy = StrategyObservation(
        opponent_position_local=jp.zeros((batch_size, 2)),
        opponent_velocity_local=jp.zeros((batch_size, 2)),
        arena_center_local=jp.zeros((batch_size, 2)),
        previous_command=jp.zeros((batch_size, 3)),
        role=jp.tile(jp.asarray([[1.0, 0.0]]), (batch_size, 1)),
    )
    locomotion = LocomotionObservation(
        root_linear_velocity=jp.zeros((batch_size, 3)),
        root_angular_velocity=jp.zeros((batch_size, 3)),
        projected_gravity=jp.tile(jp.asarray([[0.0, 0.0, -1.0]]), (batch_size, 1)),
        joint_position=jp.zeros((batch_size, 29)),
        joint_velocity=jp.zeros((batch_size, 29)),
        previous_action=jp.zeros((batch_size, 29)),
        phase_cosine=jp.ones((batch_size, 2)),
        phase_sine=jp.zeros((batch_size, 2)),
    )
    return strategy, locomotion


def _tree_equal(left, right) -> bool:
    comparisons = jax.tree.map(
        lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)), left, right
    )
    return all(jax.tree.leaves(comparisons))


def _tree_changed(left, right) -> bool:
    return not _tree_equal(left, right)


def main(config: Config) -> None:
    if config.batch_size <= 1:
        raise ValueError("batch_size must exceed one")
    policy = HierarchicalPolicy()
    strategy_observation, locomotion_observation = _observations(config.batch_size)
    parameters = policy.init(
        jax.random.PRNGKey(config.seed),
        strategy_observation,
        locomotion_observation,
    )["params"]
    output = policy.apply(
        {"params": parameters}, strategy_observation, locomotion_observation
    )
    valid = jp.ones((config.batch_size,), dtype=bool)
    strategy_scale = jp.broadcast_to(
        jp.asarray(policy.config.command_scale), output.strategy.location.shape
    )
    strategy_batch = StrategyBatch(
        observation=strategy_observation.as_array(),
        command=output.strategy.command,
        value=output.strategy.value,
        log_probability=tanh_normal_log_prob(
            output.strategy.command,
            output.strategy.location,
            output.strategy.log_std,
            strategy_scale,
        ),
        entropy=jp.zeros((config.batch_size,)),
        reward=jp.zeros((config.batch_size,)),
        done=jp.zeros((config.batch_size,), dtype=bool),
        valid=valid,
        role=strategy_observation.role,
        episode_id=jp.zeros((config.batch_size,), dtype=jp.int32),
        environment_id=jp.zeros((config.batch_size,), dtype=jp.int32),
    )
    locomotion_batch = LocomotionBatch(
        observation=locomotion_observation.as_array(output.strategy.command),
        action=output.locomotion.action,
        value=output.locomotion.value,
        log_probability=tanh_normal_log_prob(
            output.locomotion.action,
            output.locomotion.location,
            output.locomotion.log_std,
            jp.ones_like(output.locomotion.location),
        ),
        entropy=jp.zeros((config.batch_size,)),
        reward=jp.zeros((config.batch_size,)),
        done=jp.zeros((config.batch_size,), dtype=bool),
        valid=valid,
        command_tracking_error=jp.zeros((config.batch_size, 3)),
        episode_id=jp.zeros((config.batch_size,), dtype=jp.int32),
        environment_id=jp.zeros((config.batch_size,), dtype=jp.int32),
    )
    advantages = jp.linspace(-1.0, 1.0, config.batch_size)
    strategy_targets = AdvantageTargets(
        advantages=advantages,
        returns=output.strategy.value + 0.25,
    )
    locomotion_targets = AdvantageTargets(
        advantages=-advantages,
        returns=output.locomotion.value - 0.25,
    )
    updater = HierarchicalPpoUpdater(
        HierarchicalUpdateConfig(
            strategy_update_interval=config.strategy_update_interval,
            locomotion_update_interval=config.locomotion_update_interval,
        )
    )
    state_0 = updater.initialize(parameters)
    frozen_opponent = jax.tree.map(lambda x: np.asarray(x).copy(), parameters)
    update = jax.jit(updater.update)
    state_1, metrics_0 = update(
        state_0,
        strategy_batch,
        strategy_targets,
        locomotion_batch,
        locomotion_targets,
    )
    state_2, metrics_1 = update(
        state_1,
        strategy_batch,
        strategy_targets,
        locomotion_batch,
        locomotion_targets,
    )
    state_3, metrics_2 = update(
        state_2,
        strategy_batch,
        strategy_targets,
        locomotion_batch,
        locomotion_targets,
    )
    schedule = [
        [bool(metrics_0.strategy_updated), bool(metrics_0.locomotion_updated)],
        [bool(metrics_1.strategy_updated), bool(metrics_1.locomotion_updated)],
        [bool(metrics_2.strategy_updated), bool(metrics_2.locomotion_updated)],
    ]
    metric_leaves = jax.tree.leaves((metrics_0, metrics_1, metrics_2))
    checks = {
        "all_metrics_finite": all(
            bool(np.isfinite(np.asarray(value)).all()) for value in metric_leaves
        ),
        "frozen_opponent_unchanged": _tree_equal(frozen_opponent, parameters),
        "independent_optimizer_states": (
            state_0.strategy_optimizer_state is not state_0.locomotion_optimizer_state
        ),
        "locomotion_updates_every_iteration": (
            _tree_changed(
                state_0.parameters["locomotion"], state_1.parameters["locomotion"]
            )
            and _tree_changed(
                state_1.parameters["locomotion"], state_2.parameters["locomotion"]
            )
            and _tree_changed(
                state_2.parameters["locomotion"], state_3.parameters["locomotion"]
            )
        ),
        "scheduled_update_counts": (
            int(state_3.strategy_updates) == 2
            and int(state_3.locomotion_updates) == 3
            and int(state_3.iteration) == 3
        ),
        "strategy_skips_second_iteration": (
            _tree_changed(
                state_0.parameters["strategy"], state_1.parameters["strategy"]
            )
            and _tree_equal(
                state_1.parameters["strategy"], state_2.parameters["strategy"]
            )
            and _tree_equal(
                state_1.strategy_optimizer_state,
                state_2.strategy_optimizer_state,
            )
            and _tree_changed(
                state_2.parameters["strategy"], state_3.parameters["strategy"]
            )
        ),
        "update_schedule": schedule == [[True, True], [False, True], [True, True]],
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "hierarchical_optimizer_update_schedule",
        "final_counters": {
            "iteration": int(state_3.iteration),
            "locomotion_updates": int(state_3.locomotion_updates),
            "strategy_updates": int(state_3.strategy_updates),
        },
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "schedule": schedule,
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
