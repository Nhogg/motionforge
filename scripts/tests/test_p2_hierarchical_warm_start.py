"""Prove numerical parity after migrating P2 into hierarchical locomotion."""

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
    LocomotionModule,
    LocomotionObservation,
    StrategyObservation,
    load_p2_checkpoint,
    migrate_p2_actor_parameters,
    p2_locomotion_normalization,
)


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 0
    batch_size: int = 32
    absolute_tolerance: float = 1e-6
    output: Path = Path("logs/hierarchical/p2_warm_start_parity_a.json")


def initialization_observations():
    strategy = StrategyObservation(
        opponent_position_local=jp.zeros((1, 2)),
        opponent_velocity_local=jp.zeros((1, 2)),
        arena_center_local=jp.zeros((1, 2)),
        previous_command=jp.zeros((1, 3)),
        role=jp.asarray([[1.0, 0.0]]),
    )
    locomotion = LocomotionObservation(
        root_linear_velocity=jp.zeros((1, 3)),
        root_angular_velocity=jp.zeros((1, 3)),
        projected_gravity=jp.asarray([[0.0, 0.0, -1.0]]),
        joint_position=jp.zeros((1, 29)),
        joint_velocity=jp.zeros((1, 29)),
        previous_action=jp.zeros((1, 29)),
        phase_cosine=jp.ones((1, 2)),
        phase_sine=jp.zeros((1, 2)),
    )
    return strategy, locomotion


def main(config: Config) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be positive")

    p2_network, p2_parameters = load_p2_checkpoint(config.checkpoint)
    strategy, locomotion = initialization_observations()
    hierarchy = HierarchicalPolicy()
    hierarchical_parameters = hierarchy.init(
        jax.random.PRNGKey(config.seed), strategy, locomotion
    )["params"]
    migrated_parameters = migrate_p2_actor_parameters(
        hierarchical_parameters,
        p2_parameters[1],
    )
    normalization = p2_locomotion_normalization(p2_parameters[0])

    observation_key = jax.random.PRNGKey(config.seed + 1)
    raw_observation = jax.random.normal(
        observation_key,
        (config.batch_size, 103),
    )
    legacy_logits = p2_network.policy_network.apply(
        p2_parameters[0],
        p2_parameters[1],
        {"state": raw_observation},
    )
    legacy_action = p2_network.parametric_action_distribution.mode(legacy_logits)

    locomotion_module = LocomotionModule(
        hierarchy.config.locomotion_hidden_layer_sizes
    )
    migrated_output = locomotion_module.apply(
        {"params": migrated_parameters["locomotion"]},
        normalization.normalize(raw_observation),
    )
    migrated_action = migrated_output.action

    legacy_host = np.asarray(legacy_action)
    migrated_host = np.asarray(migrated_action)
    maximum_absolute_error = float(
        np.max(np.abs(legacy_host - migrated_host))
    )
    checks = {
        "actions_finite": bool(
            np.isfinite(legacy_host).all() and np.isfinite(migrated_host).all()
        ),
        "actions_match": maximum_absolute_error <= config.absolute_tolerance,
        "actions_shape": legacy_host.shape
        == migrated_host.shape
        == (config.batch_size, 29),
        "normalization_finite": bool(
            np.isfinite(np.asarray(normalization.mean)).all()
            and np.isfinite(np.asarray(normalization.std)).all()
        ),
        "normalization_shape": normalization.mean.shape
        == normalization.std.shape
        == (103,),
        "parameter_groups_preserved": set(migrated_parameters)
        == {"locomotion", "strategy"},
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "checkpoint": str(config.checkpoint.resolve()),
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "output": str(config.output),
        },
        "experiment": "p2_hierarchical_locomotion_warm_start",
        "jax_version": jax.__version__,
        "maximum_absolute_action_error": maximum_absolute_error,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
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
