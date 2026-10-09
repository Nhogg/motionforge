"""Validate exact, corruption-detecting hierarchical checkpoint round trips."""

from __future__ import annotations

import json
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import optax
from flax.core import freeze

from motionforge.cli import run_hydra
from motionforge.training import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_VERSION,
    HierarchicalPpoUpdater,
    HierarchicalUpdateConfig,
    load_hierarchical_checkpoint,
    save_hierarchical_checkpoint,
)


@dataclass
class Config:
    seed: int = 0
    output: Path = Path("logs/hierarchical/checkpoint_roundtrip_a.json")


def _tree_equal(left, right) -> bool:
    comparisons = jax.tree.map(
        lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)), left, right
    )
    return all(jax.tree.leaves(comparisons))


def main(config: Config) -> None:
    parameters = freeze(
        {
            "strategy": {"weight": jp.asarray([1.0, -2.0])},
            "locomotion": {"weight": jp.asarray([3.0, -4.0, 5.0])},
        }
    )
    update_config = HierarchicalUpdateConfig(
        strategy_learning_rate=2e-4,
        locomotion_learning_rate=7e-5,
        strategy_update_interval=2,
        locomotion_update_interval=1,
        maximum_gradient_norm=0.75,
    )
    updater = HierarchicalPpoUpdater(update_config)
    template = updater.initialize(parameters)
    strategy_gradients = jax.tree.map(jp.ones_like, parameters["strategy"])
    locomotion_gradients = jax.tree.map(
        lambda value: -jp.ones_like(value), parameters["locomotion"]
    )
    strategy_updates, strategy_optimizer_state = updater.strategy_optimizer.update(
        strategy_gradients,
        template.strategy_optimizer_state,
        parameters["strategy"],
    )
    locomotion_updates, locomotion_optimizer_state = (
        updater.locomotion_optimizer.update(
            locomotion_gradients,
            template.locomotion_optimizer_state,
            parameters["locomotion"],
        )
    )
    trained_state = template.replace(
        parameters=freeze(
            {
                "strategy": optax.apply_updates(
                    parameters["strategy"], strategy_updates
                ),
                "locomotion": optax.apply_updates(
                    parameters["locomotion"], locomotion_updates
                ),
            }
        ),
        strategy_optimizer_state=strategy_optimizer_state,
        locomotion_optimizer_state=locomotion_optimizer_state,
        iteration=jp.asarray(7, dtype=jp.int32),
        strategy_updates=jp.asarray(4, dtype=jp.int32),
        locomotion_updates=jp.asarray(7, dtype=jp.int32),
    )

    with tempfile.TemporaryDirectory(prefix="motionforge-checkpoint-") as root:
        checkpoint = Path(root) / "000000000007"
        saved = save_hierarchical_checkpoint(
            checkpoint,
            trained_state,
            update_config=update_config,
            metadata={"seed": config.seed, "purpose": "roundtrip_test"},
        )
        restored, manifest = load_hierarchical_checkpoint(saved, template)
        duplicate_rejected = False
        try:
            save_hierarchical_checkpoint(
                checkpoint,
                trained_state,
                update_config=update_config,
            )
        except FileExistsError:
            duplicate_rejected = True

        state_path = checkpoint / "learner_state.msgpack"
        original_payload = state_path.read_bytes()
        state_path.write_bytes(original_payload + b"corrupt")
        corruption_detected = False
        try:
            load_hierarchical_checkpoint(checkpoint, template)
        except ValueError:
            corruption_detected = True

    checks = {
        "complete_state_round_trip": _tree_equal(trained_state, restored),
        "config_recorded": manifest["update_config"] == asdict(update_config),
        "corruption_detected": corruption_detected,
        "counters_recorded": (
            manifest["iteration"] == 7
            and manifest["strategy_updates"] == 4
            and manifest["locomotion_updates"] == 7
        ),
        "duplicate_checkpoint_rejected": duplicate_rejected,
        "metadata_recorded": manifest["metadata"]
        == {"seed": config.seed, "purpose": "roundtrip_test"},
        "schema_versioned": (
            manifest["schema"] == CHECKPOINT_SCHEMA
            and manifest["version"] == CHECKPOINT_VERSION
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "hierarchical_checkpoint_roundtrip",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "restored_counters": {
            "iteration": int(restored.iteration),
            "locomotion_updates": int(restored.locomotion_updates),
            "strategy_updates": int(restored.strategy_updates),
        },
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
