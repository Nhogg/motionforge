"""Run one reproducible alternating TAG self-play round.

The round selects an evader, resumes and trains the current pursuer, snapshots
the result, then selects a pursuer, resumes and trains the current evader, and
snapshots that result. Every command, selection, checkpoint, and stage status
is persisted to a JSON manifest. Dry-run mode validates and records the plan
without starting GPU training or changing the policy population.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from motionforge.cli import run_hydra
from motionforge.self_play import save_policy_snapshot, select_opponent


@dataclass
class Config:
    population_dir: Path = Path("logs/p8/population")
    current_pursuer_checkpoint: Path = Path("???")
    current_evader_checkpoint: Path = Path("???")
    current_pursuer_generation: int | None = None
    current_evader_generation: int | None = None
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    round_index: int = 1
    seed: int = 0
    current_opponent_probability: float = 0.5
    slope_degrees: float = 0.0
    slope_curriculum_degrees: float = 0.0
    fixed_noise_std: float = 0.2
    num_timesteps: int = 131_072
    num_envs: int = 128
    num_eval_envs: int = 16
    num_evals: int = 2
    wandb_mode: str = "disabled"
    output_dir: Path = Path("logs/p8/rounds/round_0001")
    dry_run: bool = True


def _config_dict(config: Config) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _training_command(
    *,
    script: str,
    output_dir: Path,
    restore_checkpoint: Path,
    opponent_name: str,
    opponent_checkpoint: str,
    opponent_noise_name: str,
    opponent_noise: float,
    config: Config,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        script,
        f"locomotion_checkpoint={config.locomotion_checkpoint}",
        f"restore_checkpoint={restore_checkpoint}",
        f"{opponent_name}={opponent_checkpoint}",
        f"{opponent_noise_name}={opponent_noise}",
        f"fixed_noise_std={config.fixed_noise_std}",
        f"slope_degrees={config.slope_degrees}",
        f"slope_curriculum_degrees={config.slope_curriculum_degrees}",
        f"num_timesteps={config.num_timesteps}",
        f"num_envs={config.num_envs}",
        f"num_eval_envs={config.num_eval_envs}",
        f"num_evals={config.num_evals}",
        f"seed={seed}",
        f"wandb_mode={config.wandb_mode}",
        f"wandb_group=p8_self_play_round_{config.round_index:04d}",
        f"output_dir={output_dir}",
    ]


def _final_checkpoint(training_dir: Path) -> Path:
    summary_path = training_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("passed"):
        raise RuntimeError(f"training did not pass: {summary_path}")
    entries = summary.get("checkpoint_entries", [])
    if not entries:
        raise RuntimeError(f"training produced no checkpoint: {summary_path}")
    checkpoint = training_dir / "checkpoints" / entries[-1]
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"final checkpoint does not exist: {checkpoint}")
    return checkpoint.resolve()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(config: Config) -> None:
    if config.round_index <= 0:
        raise ValueError("round_index must be positive")
    if config.seed < 0:
        raise ValueError("seed must be nonnegative")
    pursuer_generation = (
        config.round_index - 1
        if config.current_pursuer_generation is None
        else config.current_pursuer_generation
    )
    evader_generation = (
        config.round_index - 1
        if config.current_evader_generation is None
        else config.current_evader_generation
    )
    if not 0 <= pursuer_generation < config.round_index:
        raise ValueError("current_pursuer_generation must precede round_index")
    if not 0 <= evader_generation < config.round_index:
        raise ValueError("current_evader_generation must precede round_index")
    if config.num_timesteps <= 0 or config.num_envs <= 0:
        raise ValueError("training sizes must be positive")
    if config.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {config.output_dir}")

    pursuer_checkpoint = config.current_pursuer_checkpoint.resolve()
    evader_checkpoint = config.current_evader_checkpoint.resolve()
    for checkpoint in (pursuer_checkpoint, evader_checkpoint):
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"current checkpoint does not exist: {checkpoint}")

    evader_selection = select_opponent(
        population_dir=config.population_dir,
        current_checkpoint=evader_checkpoint,
        role="evader",
        current_generation=evader_generation,
        current_fixed_noise_std=config.fixed_noise_std,
        current_probability=config.current_opponent_probability,
        seed=config.seed,
    )
    pursuer_training_dir = config.output_dir.resolve() / "pursuer_training"
    pursuer_command = _training_command(
        script="scripts/training/train_tag_pursuer.py",
        output_dir=pursuer_training_dir,
        restore_checkpoint=pursuer_checkpoint,
        opponent_name="evader_checkpoint",
        opponent_checkpoint=evader_selection.checkpoint,
        opponent_noise_name="evader_fixed_noise_std",
        opponent_noise=evader_selection.fixed_noise_std,
        config=config,
        seed=config.seed,
    )
    manifest: dict[str, Any] = {
        "config": _config_dict(config),
        "experiment": "p8_tag_self_play_round",
        "passed": False,
        "stages": {
            "evader_selection": asdict(evader_selection),
            "pursuer_training": {"command": pursuer_command, "status": "planned"},
        },
    }
    config.output_dir.mkdir(parents=True)
    manifest_path = config.output_dir / "round_manifest.json"

    if config.dry_run:
        manifest["dry_run"] = True
        manifest["passed"] = True
        _write_manifest(manifest_path, manifest)
        print(json.dumps(manifest, sort_keys=True))
        return

    _write_manifest(manifest_path, manifest)
    subprocess.run(pursuer_command, check=True)
    trained_pursuer = _final_checkpoint(pursuer_training_dir)
    pursuer_snapshot = save_policy_snapshot(
        population_dir=config.population_dir,
        source_checkpoint=trained_pursuer,
        role="pursuer",
        generation=config.round_index,
        seed=config.seed,
        fixed_noise_std=config.fixed_noise_std,
    )
    manifest["stages"]["pursuer_training"].update(
        {"checkpoint": str(trained_pursuer), "status": "passed"}
    )
    manifest["stages"]["pursuer_snapshot"] = asdict(pursuer_snapshot)
    _write_manifest(manifest_path, manifest)

    pursuer_selection = select_opponent(
        population_dir=config.population_dir,
        current_checkpoint=trained_pursuer,
        role="pursuer",
        current_generation=config.round_index,
        current_fixed_noise_std=config.fixed_noise_std,
        current_probability=config.current_opponent_probability,
        seed=config.seed + 1,
    )
    evader_training_dir = config.output_dir.resolve() / "evader_training"
    evader_command = _training_command(
        script="scripts/training/train_tag_evader.py",
        output_dir=evader_training_dir,
        restore_checkpoint=evader_checkpoint,
        opponent_name="pursuer_checkpoint",
        opponent_checkpoint=pursuer_selection.checkpoint,
        opponent_noise_name="pursuer_fixed_noise_std",
        opponent_noise=pursuer_selection.fixed_noise_std,
        config=config,
        seed=config.seed + 1,
    )
    manifest["stages"]["pursuer_selection"] = asdict(pursuer_selection)
    manifest["stages"]["evader_training"] = {
        "command": evader_command,
        "status": "planned",
    }
    _write_manifest(manifest_path, manifest)
    subprocess.run(evader_command, check=True)
    trained_evader = _final_checkpoint(evader_training_dir)
    evader_snapshot = save_policy_snapshot(
        population_dir=config.population_dir,
        source_checkpoint=trained_evader,
        role="evader",
        generation=config.round_index,
        seed=config.seed + 1,
        fixed_noise_std=config.fixed_noise_std,
    )
    manifest["stages"]["evader_training"].update(
        {"checkpoint": str(trained_evader), "status": "passed"}
    )
    manifest["stages"]["evader_snapshot"] = asdict(evader_snapshot)
    manifest["dry_run"] = False
    manifest["passed"] = True
    _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
