"""Validate the bounded local-frame command produced by the scripted pursuer."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.policies import (
    ScriptedPursuerConfig,
    scripted_pursuer_command,
)


@dataclass
class Config:
    seed: int = 0
    linear_gain: float = 0.8
    yaw_gain: float = 1.5
    tag_radius: float = 0.35
    maximum_forward_speed: float = 0.8
    maximum_lateral_speed: float = 0.4
    maximum_yaw_rate: float = 1.0
    output: Path = Path("logs/p4/tag_scripted_pursuer_a.json")


def main(config: Config) -> None:
    policy_config = ScriptedPursuerConfig(
        linear_gain=config.linear_gain,
        yaw_gain=config.yaw_gain,
        tag_radius=config.tag_radius,
        maximum_forward_speed=config.maximum_forward_speed,
        maximum_lateral_speed=config.maximum_lateral_speed,
        maximum_yaw_rate=config.maximum_yaw_rate,
    )
    policy = jax.jit(
        lambda relative_position: scripted_pursuer_command(
            relative_position,
            policy_config,
        )
    )

    fixture_positions = {
        "ahead": jp.asarray([1.0, 0.0]),
        "behind": jp.asarray([-1.0, 0.0]),
        "close": jp.asarray([0.1, 0.1]),
        "distant": jp.asarray([100.0, -100.0]),
        "left": jp.asarray([0.0, 1.0]),
        "right": jp.asarray([0.0, -1.0]),
    }
    commands = {
        name: np.asarray(policy(position))
        for name, position in fixture_positions.items()
    }

    command_stack = np.stack(tuple(commands.values()))
    checks = {
        "ahead_moves_forward": bool(commands["ahead"][0] > 0.0),
        "behind_moves_backward": bool(commands["behind"][0] < 0.0),
        "behind_turns": bool(
            np.isclose(
                abs(commands["behind"][2]),
                config.maximum_yaw_rate,
            )
        ),
        "close_translation_stops": bool(
            np.allclose(commands["close"][:2], 0.0)
        ),
        "commands_finite": bool(np.isfinite(command_stack).all()),
        "commands_within_limits": bool(
            np.all(np.abs(command_stack[:, 0]) <= config.maximum_forward_speed)
            and np.all(
                np.abs(command_stack[:, 1]) <= config.maximum_lateral_speed
            )
            and np.all(np.abs(command_stack[:, 2]) <= config.maximum_yaw_rate)
        ),
        "distant_command_clipped": bool(
            np.isclose(
                commands["distant"][0],
                config.maximum_forward_speed,
            )
            and np.isclose(
                commands["distant"][1],
                -config.maximum_lateral_speed,
            )
        ),
        "left_signs": bool(
            commands["left"][1] > 0.0 and commands["left"][2] > 0.0
        ),
        "output_shapes": all(command.shape == (3,) for command in commands.values()),
        "right_signs": bool(
            commands["right"][1] < 0.0 and commands["right"][2] < 0.0
        ),
    }

    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "commands": {
            name: command.tolist() for name, command in commands.items()
        },
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "tag_scripted_pursuer_command",
        "jax_version": jax.__version__,
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "seed": config.seed,
    }

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
