"""Verify that hierarchical TAG rollouts expose complete episode outcomes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.training import validate_full_episode_rollout


@dataclass
class Config:
    rollout_horizon: int = 500
    episode_duration: float = 10.0
    control_timestep: float = 0.02
    output: Path = Path("logs/hierarchical/episode_horizon_a.json")


def _rejected(**values: float) -> bool:
    try:
        validate_full_episode_rollout(**values)
    except ValueError:
        return True
    return False


def main(config: Config) -> None:
    arguments = {
        "rollout_horizon": config.rollout_horizon,
        "episode_duration": config.episode_duration,
        "control_timestep": config.control_timestep,
    }
    validate_full_episode_rollout(**arguments)
    checks = {
        "complete_episode_accepted": True,
        "short_rollout_rejected": _rejected(
            **{**arguments, "rollout_horizon": config.rollout_horizon // 5}
        ),
        "long_episode_rejected": _rejected(
            **{**arguments, "episode_duration": config.episode_duration * 2.0}
        ),
        "nonpositive_values_rejected": all(
            _rejected(**{**arguments, name: value})
            for name, value in (
                ("rollout_horizon", 0),
                ("episode_duration", 0.0),
                ("control_timestep", 0.0),
            )
        ),
    }
    result = {
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "hierarchical_full_episode_rollout_contract",
        "passed": all(checks.values()),
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
