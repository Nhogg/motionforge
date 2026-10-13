"""Validate independent pursuer and evader strategy output heads."""

from __future__ import annotations

import json
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from flax.core import freeze, unfreeze

from motionforge.cli import run_hydra
from motionforge.models import HierarchicalPolicy
from motionforge.training import (
    HierarchicalPpoUpdater,
    HierarchicalUpdateConfig,
    load_hierarchical_checkpoint,
    save_hierarchical_checkpoint,
)
from scripts.tests.test_hierarchical_policy import make_observations


@dataclass
class Config:
    seed: int = 0
    batch_size: int = 8
    output: Path = Path("logs/hierarchical/role_specific_heads_a.json")


def _tree_l1_norm(tree) -> float:
    return float(sum(jp.sum(jp.abs(leaf)) for leaf in jax.tree.leaves(tree)))


def main(config: Config) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pursuer_observation, locomotion_observation = make_observations(config.batch_size)
    evader_observation = pursuer_observation.replace(
        role=jp.tile(jp.asarray([[0.0, 1.0]]), (config.batch_size, 1))
    )
    policy = HierarchicalPolicy()
    parameters = policy.init(
        jax.random.PRNGKey(config.seed),
        pursuer_observation,
        locomotion_observation,
    )["params"]

    def role_loss(strategy_parameters, observation):
        current = dict(parameters)
        current["strategy"] = strategy_parameters
        output = policy.apply({"params": current}, observation, locomotion_observation)
        return jp.mean(jp.square(output.strategy.command - 0.25))

    pursuer_gradients = jax.grad(role_loss)(parameters["strategy"], pursuer_observation)
    evader_gradients = jax.grad(role_loss)(parameters["strategy"], evader_observation)
    pursuer_output = policy.apply(
        {"params": parameters}, pursuer_observation, locomotion_observation
    )
    evader_output = policy.apply(
        {"params": parameters}, evader_observation, locomotion_observation
    )

    parameter_dict = unfreeze(parameters)
    legacy_parameters = freeze(
        {
            "strategy": {
                "encoder": parameter_dict["strategy"]["encoder"],
                "head": parameter_dict["strategy"]["pursuer_head"],
            },
            "locomotion": parameter_dict["locomotion"],
        }
    )
    update_config = HierarchicalUpdateConfig()
    updater = HierarchicalPpoUpdater(update_config)
    with tempfile.TemporaryDirectory(prefix="motionforge-role-heads-") as root:
        checkpoint = Path(root) / "000000000000"
        save_hierarchical_checkpoint(
            checkpoint,
            updater.initialize(legacy_parameters),
            update_config=update_config,
        )
        migrated, migration_manifest = load_hierarchical_checkpoint(
            checkpoint, updater.initialize(parameters)
        )

    checks = {
        "distinct_parameter_heads": set(parameters["strategy"])
        == {"encoder", "pursuer_head", "evader_head"},
        "evader_batch_updates_evader_head": (
            _tree_l1_norm(evader_gradients["evader_head"]) > 0.0
        ),
        "evader_batch_spares_pursuer_head": (
            _tree_l1_norm(evader_gradients["pursuer_head"]) == 0.0
        ),
        "finite_outputs": bool(
            np.isfinite(np.asarray(pursuer_output.strategy.command)).all()
            and np.isfinite(np.asarray(evader_output.strategy.command)).all()
        ),
        "legacy_head_duplicated": bool(
            np.array_equal(
                np.asarray(
                    migrated.parameters["strategy"]["pursuer_head"]["command"]["kernel"]
                ),
                np.asarray(
                    migrated.parameters["strategy"]["evader_head"]["command"]["kernel"]
                ),
            )
        ),
        "legacy_migration_reported": bool(
            migration_manifest.get("legacy_strategy_head_migrated")
        ),
        "pursuer_batch_spares_evader_head": (
            _tree_l1_norm(pursuer_gradients["evader_head"]) == 0.0
        ),
        "pursuer_batch_updates_pursuer_head": (
            _tree_l1_norm(pursuer_gradients["pursuer_head"]) > 0.0
        ),
        "role_outputs_differ": not np.allclose(
            np.asarray(pursuer_output.strategy.command),
            np.asarray(evader_output.strategy.command),
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "hierarchical_role_specific_strategy_heads",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
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
