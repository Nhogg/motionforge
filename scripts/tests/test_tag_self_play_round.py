"""Audit a self-play round plan without launching training or mutating policies."""

from __future__ import annotations

import json
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.self_play import save_policy_snapshot
from scripts.self_play.run_tag_self_play_round import Config as RoundConfig
from scripts.self_play.run_tag_self_play_round import _training_command
from scripts.self_play.run_tag_self_play_round import main as run_round


@dataclass
class Config:
    output: Path = Path("logs/p8/tag_self_play_round_a.json")


def _checkpoint(root: Path, name: str) -> Path:
    checkpoint = root / name
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(name.encode())
    return checkpoint


def main(config: Config) -> None:
    with tempfile.TemporaryDirectory(prefix="motionforge_round_") as temporary:
        root = Path(temporary)
        pursuer = _checkpoint(root, "pursuer")
        evader = _checkpoint(root, "evader")
        locomotion = _checkpoint(root, "locomotion")
        population = root / "population"
        save_policy_snapshot(
            population_dir=population,
            source_checkpoint=pursuer,
            role="pursuer",
            generation=0,
            seed=0,
            fixed_noise_std=0.2,
        )
        save_policy_snapshot(
            population_dir=population,
            source_checkpoint=evader,
            role="evader",
            generation=0,
            seed=0,
            fixed_noise_std=0.2,
        )
        round_dir = root / "round"
        registry_before = (population / "population.json").read_bytes()
        round_config = RoundConfig(
            population_dir=population,
            current_pursuer_checkpoint=pursuer,
            current_evader_checkpoint=evader,
            current_pursuer_generation=0,
            current_evader_generation=0,
            locomotion_checkpoint=locomotion,
            round_index=1,
            slope_curriculum_degrees=5.0,
            evader_boundary_command_safety_enabled=True,
            output_dir=round_dir,
            dry_run=True,
        )
        run_round(round_config)
        evader_command = _training_command(
            script="scripts/training/train_tag_evader.py",
            output_dir=round_dir / "evader_training",
            restore_checkpoint=evader,
            opponent_name="pursuer_checkpoint",
            opponent_checkpoint=str(pursuer),
            opponent_noise_name="pursuer_fixed_noise_std",
            opponent_noise=0.2,
            config=round_config,
            seed=1,
        )
        manifest = json.loads(
            (round_dir / "round_manifest.json").read_text(encoding="utf-8")
        )
        command = manifest["stages"]["pursuer_training"]["command"]
        checks = {
            "dry_run_passed": manifest["passed"] and manifest["dry_run"],
            "evader_selected": (
                manifest["stages"]["evader_selection"]["role"] == "evader"
            ),
            "population_unchanged": (
                (population / "population.json").read_bytes() == registry_before
            ),
            "restore_checkpoint_planned": any(
                item == f"restore_checkpoint={pursuer.resolve()}" for item in command
            ),
            "slope_curriculum_planned": (
                "slope_curriculum_degrees=5.0" in command
            ),
            "evader_failure_penalties_planned": (
                "tag_penalty=10.0" in evader_command
                and "evader_fall_penalty=25.0" in evader_command
                and "evader_out_of_bounds_penalty=25.0" in evader_command
                and "boundary_outward_velocity_penalty_scale=5.0"
                in evader_command
                and "boundary_command_safety_enabled=True" in evader_command
            ),
            "training_not_started": not (round_dir / "pursuer_training").exists(),
        }

    result = {
        "checks": checks,
        "config": {"output": str(config.output)},
        "experiment": "p8_tag_self_play_round",
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
