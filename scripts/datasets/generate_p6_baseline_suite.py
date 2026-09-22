"""Generate and validate the final equal-budget P6 baseline dataset suite.

The suite runs the ordinary-command, aggressive-command, and terrain-curriculum
generators sequentially with one checkpoint, seed, and transition budget. It
then independently streams each JSONL artifact to verify its row count and
SHA-256 digest before writing a machine-readable suite manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.datasets import LOCOMOTION_DATASET_SCHEMA_VERSION, json_default


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    samples: int = 1_000_000
    episode_steps: int = 128
    command_hold_seconds: float = 0.50
    output_directory: Path = Path("logs/p6/final_seed0")
    manifest: Path = Path("logs/p6/final_seed0/suite_manifest.json")
    run_generators: bool = True


GENERATORS = {
    "ordinary": (
        Path("scripts/datasets/generate_random_commands.py"),
        "ordinary_random_commands",
    ),
    "aggressive": (
        Path("scripts/datasets/generate_aggressive_commands.py"),
        "aggressive_commands",
    ),
    "terrain": (
        Path("scripts/datasets/generate_terrain_curriculum.py"),
        "terrain_curriculum",
    ),
}


def validate_config(config: Config) -> None:
    if config.samples <= 0:
        raise ValueError("samples must be positive")
    if config.episode_steps <= 0:
        raise ValueError("episode_steps must be positive")
    if config.command_hold_seconds <= 0.0:
        raise ValueError("command_hold_seconds must be positive")
    if config.manifest.resolve().parent != config.output_directory.resolve():
        raise ValueError("suite manifest must be inside output_directory")


def artifact_paths(config: Config, name: str) -> tuple[Path, Path]:
    stem = f"{name}_{config.samples}_seed{config.seed}"
    return (
        config.output_directory / f"{stem}.jsonl",
        config.output_directory / f"{stem}_manifest.json",
    )


def run_generator(config: Config, name: str, script: Path) -> None:
    output, manifest = artifact_paths(config, name)
    arguments = [
        sys.executable,
        str(script),
        f"checkpoint={config.checkpoint}",
        f"seed={config.seed}",
        f"samples={config.samples}",
        f"output={output}",
        f"manifest={manifest}",
    ]
    if name == "aggressive":
        arguments.append(f"command_hold_seconds={config.command_hold_seconds}")
    if name == "terrain":
        arguments.append(f"episode_steps={config.episode_steps}")
    subprocess.run(arguments, check=True)


def inspect_jsonl(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            rows += 1
    return rows, digest.hexdigest()


def main(config: Config) -> None:
    validate_config(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    if config.run_generators:
        for name, (script, _) in GENERATORS.items():
            run_generator(config, name, script)

    datasets = {}
    for name, (_, expected_kind) in GENERATORS.items():
        output, manifest_path = artifact_paths(config, name)
        if not output.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing {name} artifact or manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows, output_sha256 = inspect_jsonl(output)
        datasets[name] = {
            "checks": {
                "dataset_manifest_passed": manifest["passed"],
                "dataset_kind": manifest["dataset_kind"] == expected_kind,
                "file_digest": output_sha256 == manifest["output_sha256"],
                "row_count": rows == config.samples == manifest["samples"],
                "schema_version": manifest["schema_version"]
                == LOCOMOTION_DATASET_SCHEMA_VERSION,
            },
            "control_timestep": manifest["control_timestep"],
            "dataset_kind": manifest["dataset_kind"],
            "manifest": str(manifest_path.resolve()),
            "output": str(output.resolve()),
            "output_bytes": output.stat().st_size,
            "output_sha256": output_sha256,
            "rows": rows,
        }

    manifests = {
        name: json.loads(Path(dataset["manifest"]).read_text(encoding="utf-8"))
        for name, dataset in datasets.items()
    }
    checkpoints = {manifest["checkpoint"] for manifest in manifests.values()}
    seeds = {manifest["seed"] for manifest in manifests.values()}
    control_timesteps = {
        manifest["control_timestep"] for manifest in manifests.values()
    }
    schema_versions = {manifest["schema_version"] for manifest in manifests.values()}
    checks = {
        "aggressive_coverage": all(
            manifests["aggressive"]["checks"][name]
            for name in (
                "all_maneuvers_present",
                "all_phases_present",
                "high_yaw_present",
            )
        ),
        "all_dataset_checks": all(
            all(dataset["checks"].values()) for dataset in datasets.values()
        ),
        "common_checkpoint": checkpoints == {str(config.checkpoint.resolve())},
        "common_control_timestep": len(control_timesteps) == 1,
        "common_schema": schema_versions == {LOCOMOTION_DATASET_SCHEMA_VERSION},
        "common_seed": seeds == {config.seed},
        "equal_sample_budget": {dataset["rows"] for dataset in datasets.values()}
        == {config.samples},
        "terrain_coverage": all(
            manifests["terrain"]["checks"][name]
            for name in ("all_curriculum_stages_present", "both_terrains_present")
        ),
    }
    result = {
        "checks": checks,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "datasets": datasets,
        "experiment": "p6_equal_budget_baseline_suite",
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "schema_version": LOCOMOTION_DATASET_SCHEMA_VERSION,
        "total_bytes": sum(dataset["output_bytes"] for dataset in datasets.values()),
        "total_rows": sum(dataset["rows"] for dataset in datasets.values()),
    }
    config.manifest.write_text(
        json.dumps(result, default=json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, default=json_default, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
