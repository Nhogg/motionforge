"""Select one current or historical opponent for a reproducible training job.

The selector reads a self-play population, filters it by opponent role, and
uses the supplied seed and current-opponent probability to write one exact
selection to JSON. It does not start training or mutate the population.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.self_play import select_opponent


@dataclass
class Config:
    population_dir: Path = Path("logs/p8/population")
    current_checkpoint: Path = Path("???")
    role: str = "evader"
    current_generation: int = 0
    current_fixed_noise_std: float = 0.2
    current_probability: float = 0.5
    seed: int = 0
    output: Path = Path("logs/p8/opponent_selection.json")


def main(config: Config) -> None:
    selection = select_opponent(
        population_dir=config.population_dir,
        current_checkpoint=config.current_checkpoint,
        role=config.role,
        current_generation=config.current_generation,
        current_fixed_noise_std=config.current_fixed_noise_std,
        current_probability=config.current_probability,
        seed=config.seed,
    )
    result = {
        "config": {
            **asdict(config),
            "current_checkpoint": str(config.current_checkpoint),
            "output": str(config.output),
            "population_dir": str(config.population_dir),
        },
        "experiment": "p8_select_tag_opponent",
        "passed": True,
        "selection": asdict(selection),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
