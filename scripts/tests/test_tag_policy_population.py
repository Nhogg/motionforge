"""Audit immutable snapshot copies and deterministic population registration."""

from __future__ import annotations

import json
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.policies.tag_learned import checkpoint_digest
from motionforge.self_play import save_policy_snapshot


@dataclass
class Config:
    output: Path = Path("logs/p8/tag_policy_population_a.json")


def main(config: Config) -> None:
    with tempfile.TemporaryDirectory(prefix="motionforge_population_") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        (source / "weights").write_bytes(b"generation-zero")
        population = root / "population"
        snapshot = save_policy_snapshot(
            population_dir=population,
            source_checkpoint=source,
            role="pursuer",
            generation=0,
            seed=7,
            fixed_noise_std=0.2,
        )
        snapshot_path = population / snapshot.checkpoint
        original_digest = snapshot.checkpoint_digest
        (source / "weights").write_bytes(b"source-was-mutated")

        duplicate_rejected = False
        try:
            save_policy_snapshot(
                population_dir=population,
                source_checkpoint=source,
                role="pursuer",
                generation=0,
                seed=7,
                fixed_noise_std=0.2,
            )
        except FileExistsError:
            duplicate_rejected = True

        registry = json.loads(
            (population / "population.json").read_text(encoding="utf-8")
        )
        checks = {
            "checkpoint_copied": snapshot_path.is_dir(),
            "copy_is_immutable_from_source": (
                checkpoint_digest(snapshot_path) == original_digest
            ),
            "digest_recorded": len(original_digest) == 64,
            "duplicate_generation_rejected": duplicate_rejected,
            "registry_contains_snapshot": registry["snapshots"] == [asdict(snapshot)],
            "registry_versioned": registry["registry_version"] == 1,
            "stable_snapshot_id": snapshot.snapshot_id
            == f"pursuer_g0000_{original_digest[:12]}",
        }

    result = {
        "checks": checks,
        "config": {"output": str(config.output)},
        "experiment": "p8_tag_policy_population",
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
