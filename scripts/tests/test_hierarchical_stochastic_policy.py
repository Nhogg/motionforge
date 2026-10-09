"""Verify PPO-ready stochastic hierarchical action distributions."""

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
    diagonal_normal_entropy,
    tanh_normal_log_prob,
    tanh_normal_sample_and_log_prob,
)


@dataclass
class Config:
    seed: int = 0
    batch_size: int = 32
    output: Path = Path("logs/hierarchical/stochastic_policy_a.json")


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


def main(config: Config) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    parameter_key, strategy_key, locomotion_key, other_key = jax.random.split(
        jax.random.PRNGKey(config.seed), 4
    )
    strategy_observation, locomotion_observation = _observations(config.batch_size)
    policy = HierarchicalPolicy()
    parameters = policy.init(
        parameter_key, strategy_observation, locomotion_observation
    )["params"]
    deterministic = policy.apply(
        {"params": parameters}, strategy_observation, locomotion_observation
    )
    strategy_scale = jp.broadcast_to(
        jp.asarray(policy.config.command_scale), deterministic.strategy.location.shape
    )
    strategy_action, strategy_log_probability = tanh_normal_sample_and_log_prob(
        strategy_key,
        deterministic.strategy.location,
        deterministic.strategy.log_std,
        strategy_scale,
    )
    repeated_action, repeated_log_probability = tanh_normal_sample_and_log_prob(
        strategy_key,
        deterministic.strategy.location,
        deterministic.strategy.log_std,
        strategy_scale,
    )
    other_action, _ = tanh_normal_sample_and_log_prob(
        other_key,
        deterministic.strategy.location,
        deterministic.strategy.log_std,
        strategy_scale,
    )
    locomotion_action, locomotion_log_probability = (
        tanh_normal_sample_and_log_prob(
            locomotion_key,
            deterministic.locomotion.location,
            deterministic.locomotion.log_std,
            jp.ones_like(deterministic.locomotion.location),
        )
    )
    recomputed_strategy_log_probability = tanh_normal_log_prob(
        strategy_action,
        deterministic.strategy.location,
        deterministic.strategy.log_std,
        strategy_scale,
    )
    strategy_entropy = diagonal_normal_entropy(deterministic.strategy.log_std)
    locomotion_entropy = diagonal_normal_entropy(deterministic.locomotion.log_std)

    strategy_host = np.asarray(strategy_action)
    locomotion_host = np.asarray(locomotion_action)
    checks = {
        "deterministic_means_bounded": bool(
            np.all(np.abs(np.asarray(deterministic.strategy.command))
                   <= np.asarray(policy.config.command_scale) + 1e-6)
            and np.all(np.abs(np.asarray(deterministic.locomotion.action)) <= 1.0)
        ),
        "distribution_parameters_trainable": (
            parameters["strategy"]["head"]["log_std"].shape == (3,)
            and parameters["locomotion"]["motor_head"]["log_std"].shape
            == (29,)
        ),
        "entropy_shapes": (
            strategy_entropy.shape == (config.batch_size,)
            and locomotion_entropy.shape == (config.batch_size,)
        ),
        "log_probabilities_finite": bool(
            np.isfinite(np.asarray(strategy_log_probability)).all()
            and np.isfinite(np.asarray(locomotion_log_probability)).all()
        ),
        "log_probability_recomputes": bool(
            np.allclose(
                np.asarray(strategy_log_probability),
                np.asarray(recomputed_strategy_log_probability),
                atol=1e-5,
            )
        ),
        "sampled_actions_bounded": bool(
            np.all(np.abs(strategy_host)
                   <= np.asarray(policy.config.command_scale) + 1e-6)
            and np.all(np.abs(locomotion_host) <= 1.0)
        ),
        "sampling_key_controls_actions": bool(
            np.array_equal(strategy_host, np.asarray(repeated_action))
            and np.array_equal(
                np.asarray(strategy_log_probability),
                np.asarray(repeated_log_probability),
            )
            and not np.array_equal(strategy_host, np.asarray(other_action))
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "hierarchical_stochastic_policy",
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
