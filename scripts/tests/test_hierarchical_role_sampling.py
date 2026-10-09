"""Verify deterministic and balanced learner-role sampling by segment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.rollout import sample_learner_is_pursuer


@dataclass
class Config:
    seed: int = 0
    samples: int = 1000
    pursuer_probability: float = 0.5
    tolerance: float = 0.05
    output: Path = Path("logs/hierarchical/role_sampling_a.json")


def main(config: Config) -> None:
    if config.samples <= 0:
        raise ValueError("samples must be positive")
    roles = [
        sample_learner_is_pursuer(
            seed=config.seed + offset,
            probability=config.pursuer_probability,
        )
        for offset in range(config.samples)
    ]
    repeat = [
        sample_learner_is_pursuer(
            seed=config.seed + offset,
            probability=config.pursuer_probability,
        )
        for offset in range(config.samples)
    ]
    observed_probability = sum(roles) / config.samples
    checks = {
        "both_roles_sampled": any(roles) and not all(roles),
        "deterministic": roles == repeat,
        "frequency_within_tolerance": abs(
            observed_probability - config.pursuer_probability
        )
        <= config.tolerance,
        "probability_one_is_pursuer": all(
            sample_learner_is_pursuer(seed=seed, probability=1.0)
            for seed in range(10)
        ),
        "probability_zero_is_evader": not any(
            sample_learner_is_pursuer(seed=seed, probability=0.0)
            for seed in range(10)
        ),
    }
    result = {
        "checks": checks,
        "experiment": "hierarchical_learner_role_sampling",
        "observed_pursuer_probability": observed_probability,
        "passed": all(checks.values()),
        "samples": config.samples,
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
