"""Validate independent PPO losses and the hierarchy gradient boundary."""

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
from motionforge.training import locomotion_ppo_loss, strategy_ppo_loss


@dataclass
class Config:
    seed: int = 0
    batch_size: int = 8
    gradient_tolerance: float = 1e-10
    output: Path = Path("logs/hierarchical/ppo_losses_a.json")


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
        projected_gravity=jp.tile(
            jp.asarray([[0.0, 0.0, -1.0]]), (batch_size, 1)
        ),
        joint_position=jp.zeros((batch_size, 29)),
        joint_velocity=jp.zeros((batch_size, 29)),
        previous_action=jp.zeros((batch_size, 29)),
        phase_cosine=jp.ones((batch_size, 2)),
        phase_sine=jp.zeros((batch_size, 2)),
    )
    return strategy, locomotion


def _maximum_absolute(tree) -> float:
    return max(float(np.max(np.abs(np.asarray(leaf)))) for leaf in jax.tree.leaves(tree))


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
    strategy_log_probability = tanh_normal_log_prob(
        output.strategy.command,
        output.strategy.location,
        output.strategy.log_std,
        strategy_scale,
    )
    locomotion_log_probability = tanh_normal_log_prob(
        output.locomotion.action,
        output.locomotion.location,
        output.locomotion.log_std,
        jp.ones_like(output.locomotion.location),
    )
    strategy_batch = StrategyBatch(
        observation=strategy_observation.as_array(),
        command=output.strategy.command,
        value=output.strategy.value,
        log_probability=strategy_log_probability,
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
        log_probability=locomotion_log_probability,
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
    (strategy_loss, strategy_metrics), strategy_gradient = jax.value_and_grad(
        strategy_ppo_loss, has_aux=True
    )(parameters["strategy"], strategy_batch, strategy_targets)
    (locomotion_loss, locomotion_metrics), locomotion_gradient = (
        jax.value_and_grad(locomotion_ppo_loss, has_aux=True)(
            parameters["locomotion"], locomotion_batch, locomotion_targets
        )
    )

    def composed_motor_loss(all_parameters):
        policy_output = policy.apply(
            {"params": all_parameters},
            strategy_observation,
            locomotion_observation,
        )
        return jp.mean(jp.square(policy_output.locomotion.action))

    composed_gradient = jax.grad(composed_motor_loss)(parameters)
    strategy_gradient_max = _maximum_absolute(strategy_gradient)
    locomotion_gradient_max = _maximum_absolute(locomotion_gradient)
    blocked_strategy_gradient_max = _maximum_absolute(
        composed_gradient["strategy"]
    )
    composed_locomotion_gradient_max = _maximum_absolute(
        composed_gradient["locomotion"]
    )
    metric_values = jax.tree.leaves((strategy_metrics, locomotion_metrics))
    checks = {
        "composed_motor_gradient_blocked": (
            blocked_strategy_gradient_max <= config.gradient_tolerance
            and composed_locomotion_gradient_max > config.gradient_tolerance
        ),
        "locomotion_ppo_gradient_nonzero": (
            locomotion_gradient_max > config.gradient_tolerance
        ),
        "losses_finite": bool(
            np.isfinite(float(strategy_loss))
            and np.isfinite(float(locomotion_loss))
        ),
        "metrics_finite": all(
            bool(np.isfinite(np.asarray(value)).all()) for value in metric_values
        ),
        "strategy_ppo_gradient_nonzero": (
            strategy_gradient_max > config.gradient_tolerance
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "hierarchical_independent_ppo_losses",
        "gradient_maximums": {
            "composed_locomotion": composed_locomotion_gradient_max,
            "locomotion_ppo": locomotion_gradient_max,
            "strategy_blocked_from_motor": blocked_strategy_gradient_max,
            "strategy_ppo": strategy_gradient_max,
        },
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
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
