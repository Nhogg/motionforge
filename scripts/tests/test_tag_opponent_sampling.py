"""Audit seeded selection between current and role-filtered historical policies."""

from __future__ import annotations

import json
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.self_play import save_policy_snapshot, select_opponent


@dataclass
class Config:
    output: Path = Path("logs/p8/tag_opponent_sampling_a.json")


def _checkpoint(parent: Path, name: str) -> Path:
    checkpoint = parent / name
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(name.encode())
    return checkpoint


def main(config: Config) -> None:
    with tempfile.TemporaryDirectory(prefix="motionforge_sampling_") as temporary:
        root = Path(temporary)
        population = root / "population"
        current = _checkpoint(root, "current_evader")
        historical_a = _checkpoint(root, "historical_evader_a")
        historical_b = _checkpoint(root, "historical_evader_b")
        pursuer = _checkpoint(root, "historical_pursuer")
        for generation, checkpoint in enumerate((historical_a, historical_b)):
            save_policy_snapshot(
                population_dir=population,
                source_checkpoint=checkpoint,
                role="evader",
                generation=generation,
                seed=generation,
                fixed_noise_std=0.2,
            )
        save_policy_snapshot(
            population_dir=population,
            source_checkpoint=pursuer,
            role="pursuer",
            generation=0,
            seed=0,
            fixed_noise_std=0.2,
        )

        arguments = {
            "population_dir": population,
            "current_checkpoint": current,
            "role": "evader",
            "current_generation": 2,
            "current_fixed_noise_std": 0.2,
            "seed": 11,
        }
        current_selection = select_opponent(
            **arguments, current_probability=1.0
        )
        historical_selection = select_opponent(
            **arguments, current_probability=0.0
        )
        repeated_selection = select_opponent(
            **arguments, current_probability=0.0
        )
        checks = {
            "current_can_be_forced": current_selection.source == "current",
            "historical_can_be_forced": (
                historical_selection.source == "historical"
            ),
            "historical_role_filtered": historical_selection.role == "evader",
            "historical_snapshot_identified": (
                historical_selection.snapshot_id is not None
            ),
            "seeded_selection_repeatable": (
                historical_selection == repeated_selection
            ),
            "selection_records_probability": (
                historical_selection.current_probability == 0.0
            ),
        }

    result = {
        "checks": checks,
        "config": {"output": str(config.output)},
        "experiment": "p8_tag_opponent_sampling",
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
