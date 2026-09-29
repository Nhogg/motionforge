"""Copy one trained TAG checkpoint into a versioned self-play population.

Inputs are a checkpoint directory plus role, generation, seed, and policy
noise contract. The command writes an immutable checkpoint copy and updates
the population's JSON registry; an existing role/generation is never replaced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.self_play import save_policy_snapshot


@dataclass
class Config:
    population_dir: Path = Path("logs/p8/population")
    checkpoint: Path = Path("???")
    role: str = "pursuer"
    generation: int = 0
    seed: int = 0
    fixed_noise_std: float = 0.2


def main(config: Config) -> None:
    snapshot = save_policy_snapshot(
        population_dir=config.population_dir,
        source_checkpoint=config.checkpoint,
        role=config.role,
        generation=config.generation,
        seed=config.seed,
        fixed_noise_std=config.fixed_noise_std,
    )
    result = {
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "population_dir": str(config.population_dir),
        },
        "experiment": "p8_save_tag_policy_snapshot",
        "passed": True,
        "snapshot": asdict(snapshot),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
