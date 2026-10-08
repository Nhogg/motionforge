"""Validate the first role-conditioned hierarchical policy contract."""

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
    LOCOMOTION_ACTION_SIZE,
    LOCOMOTION_OBSERVATION_SIZE,
    STRATEGY_ACTION_SIZE,
    STRATEGY_OBSERVATION_SIZE,
    HierarchicalPolicy,
    LocomotionObservation,
    StrategyObservation,
)


@dataclass
class Config:
    seed: int = 0
    batch_size: int = 8
    output: Path = Path("logs/hierarchical/policy_contract_a.json")


def make_observations(batch_size: int):
    strategy = StrategyObservation(
        opponent_position_local=jp.ones((batch_size, 2)),
        opponent_velocity_local=jp.full((batch_size, 2), 0.25),
        arena_center_local=jp.full((batch_size, 2), -0.5),
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


def tree_l1_norm(tree) -> float:
    return float(sum(jp.sum(jp.abs(leaf)) for leaf in jax.tree.leaves(tree)))


def main(config: Config) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model = HierarchicalPolicy()
    strategy, locomotion = make_observations(config.batch_size)
    parameters = model.init(
        jax.random.PRNGKey(config.seed), strategy, locomotion
    )["params"]
    apply = jax.jit(model.apply)
    output = apply({"params": parameters}, strategy, locomotion)

    evader_strategy = strategy.replace(
        role=jp.tile(jp.asarray([[0.0, 1.0]]), (config.batch_size, 1))
    )
    evader_output = apply({"params": parameters}, evader_strategy, locomotion)

    def motor_only_loss(current_parameters):
        current_output = model.apply(
            {"params": current_parameters}, strategy, locomotion
        )
        return jp.mean(jp.square(current_output.locomotion.action - 0.25))

    def strategy_only_loss(current_parameters):
        current_output = model.apply(
            {"params": current_parameters}, strategy, locomotion
        )
        return jp.mean(jp.square(current_output.strategy.command - 0.25))

    motor_gradients = jax.grad(motor_only_loss)(parameters)
    strategy_gradient_from_motor = tree_l1_norm(motor_gradients["strategy"])
    locomotion_gradient_from_motor = tree_l1_norm(motor_gradients["locomotion"])
    strategy_gradients = jax.grad(strategy_only_loss)(parameters)
    strategy_gradient_from_strategy = tree_l1_norm(strategy_gradients["strategy"])
    locomotion_gradient_from_strategy = tree_l1_norm(
        strategy_gradients["locomotion"]
    )

    strategy_array = strategy.as_array()
    locomotion_array = locomotion.as_array(output.strategy.command)
    command = np.asarray(output.strategy.command)
    action = np.asarray(output.locomotion.action)
    checks = {
        "action_bounded": bool(np.all(action >= -1.0) and np.all(action <= 1.0)),
        "action_finite": bool(np.isfinite(action).all()),
        "action_shape": action.shape
        == (config.batch_size, LOCOMOTION_ACTION_SIZE),
        "command_bounded": bool(
            np.all(np.abs(command[:, 0]) <= 1.0)
            and np.all(np.abs(command[:, 1]) <= 0.5)
            and np.all(np.abs(command[:, 2]) <= 1.0)
        ),
        "command_finite": bool(np.isfinite(command).all()),
        "command_shape": command.shape
        == (config.batch_size, STRATEGY_ACTION_SIZE),
        "command_at_p2_indices": bool(
            np.array_equal(
                np.asarray(locomotion_array[:, 9:12]),
                command,
            )
        ),
        "locomotion_gradient_from_motor_nonzero": (
            locomotion_gradient_from_motor > 0.0
        ),
        "locomotion_gradient_from_strategy_zero": (
            locomotion_gradient_from_strategy == 0.0
        ),
        "parameter_groups": set(parameters) == {"locomotion", "strategy"},
        "locomotion_observation_shape": locomotion_array.shape
        == (config.batch_size, LOCOMOTION_OBSERVATION_SIZE),
        "locomotion_value_shape": np.asarray(output.locomotion.value).shape
        == (config.batch_size,),
        "role_conditioning": not np.allclose(
            command, np.asarray(evader_output.strategy.command)
        ),
        "strategy_gradient_from_motor_stopped": (
            strategy_gradient_from_motor == 0.0
        ),
        "strategy_gradient_from_strategy_nonzero": (
            strategy_gradient_from_strategy > 0.0
        ),
        "strategy_observation_shape": strategy_array.shape
        == (config.batch_size, STRATEGY_OBSERVATION_SIZE),
        "strategy_value_shape": np.asarray(output.strategy.value).shape
        == (config.batch_size,),
    }
    result = {
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "hierarchical_policy_contract",
        "gradient_norms": {
            "locomotion_from_motor_loss": locomotion_gradient_from_motor,
            "locomotion_from_strategy_loss": locomotion_gradient_from_strategy,
            "strategy_from_motor_loss": strategy_gradient_from_motor,
            "strategy_from_strategy_loss": strategy_gradient_from_strategy,
        },
        "jax_version": jax.__version__,
        "observation_sizes": {
            "locomotion": LOCOMOTION_OBSERVATION_SIZE,
            "strategy": STRATEGY_OBSERVATION_SIZE,
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
