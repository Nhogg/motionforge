"""test_g1_command_sampling.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>

Description:
    Validate MotionForge's structured G1 command distribution.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import numpy as np
import tyro

from motionforge.envs.g1_standing import G1StandingJoystick


@dataclass(frozen=True)
class Config:
    seed: int = 0
    samples: int = 16_384

    standing_probability: float = 0.30
    pure_x_probability: float = 0.15
    pure_y_probability: float = 0.10
    pure_yaw_probability: float = 0.20
    mixed_probability: float = 0.25

    playground_root: Path = Path("../mujoco_playground")
    output: Path = Path("logs/p2/g1_structured_command_sampling_a.json")


def git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_smoke_test(config: Config) -> dict[str, Any]:
    if config.samples <= 0:
        raise ValueError("--samples must be positive")

    expected_probabilities = {
        "standing": config.standing_probability,
        "pure_x": config.pure_x_probability,
        "pure_y": config.pure_y_probability,
        "pure_yaw": config.pure_yaw_probability,
        "mixed": config.mixed_probability,
    }

    if any(probability < 0.0 for probability in expected_probabilities.values()):
        raise ValueError("Mode probabilities must be non-negative")

    probability_sum = sum(expected_probabilities.values())
    if abs(probability_sum - 1.0) > 1e-9:
        raise ValueError(f"Mode probabilities must sum to one; got {probability_sum}")

    environment = G1StandingJoystick(
        config_overrides={
            "standing_probability": config.standing_probability,
            "pure_x_probability": config.pure_x_probability,
            "pure_y_probability": config.pure_y_probability,
            "pure_yaw_probability": config.pure_yaw_probability,
            "mixed_probability": config.mixed_probability,
        }
    )

    reset_key = jax.random.PRNGKey(config.seed)
    sampling_key = jax.random.PRNGKey(config.seed + 1)

    state = environment.reset(reset_key)

    keys = jax.random.split(sampling_key, config.samples)
    commands = np.asarray(jax.device_get(jax.vmap(environment.sample_command)(keys)))

    x_nonzero = commands[:, 0] != 0.0
    y_nonzero = commands[:, 1] != 0.0
    yaw_nonzero = commands[:, 2] != 0.0

    mode_masks = {
        "standing": ~x_nonzero & ~y_nonzero & ~yaw_nonzero,
        "pure_x": x_nonzero & ~y_nonzero & ~yaw_nonzero,
        "pure_y": ~x_nonzero & y_nonzero & ~yaw_nonzero,
        "pure_yaw": ~x_nonzero & ~y_nonzero & yaw_nonzero,
        "mixed": x_nonzero & y_nonzero & yaw_nonzero,
    }

    classified = np.zeros(config.samples, dtype=bool)
    for mask in mode_masks.values():
        if np.any(classified & mask):
            raise RuntimeError("Command modes overlap")
        classified |= mask

    mode_counts = {name: int(mask.sum()) for name, mask in mode_masks.items()}
    observed_probabilities = {
        name: count / config.samples for name, count in mode_counts.items()
    }

    probability_tolerances = {
        name: max(
            0.015,
            5.0 * math.sqrt(probability * (1.0 - probability) / config.samples),
        )
        for name, probability in expected_probabilities.items()
    }

    probability_checks = {
        name: (
            abs(observed_probabilities[name] - expected_probabilities[name])
            <= probability_tolerances[name]
        )
        for name in expected_probabilities
    }

    x_limits = environment._config.lin_vel_x
    y_limits = environment._config.lin_vel_y
    yaw_limits = environment._config.ang_vel_yaw

    commands_in_range = bool(
        np.all(commands[:, 0] >= x_limits[0])
        and np.all(commands[:, 0] <= x_limits[1])
        and np.all(commands[:, 1] >= y_limits[0])
        and np.all(commands[:, 1] <= y_limits[1])
        and np.all(commands[:, 2] >= yaw_limits[0])
        and np.all(commands[:, 2] <= yaw_limits[1])
    )

    motionforge_root = Path(__file__).resolve().parents[2]
    playground_root = config.playground_root.resolve()

    checks = {
        "all_commands_classified": bool(classified.all()),
        "commands_finite": bool(np.isfinite(commands).all()),
        "commands_in_range": commands_in_range,
        "every_mode_observed": all(count > 0 for count in mode_counts.values()),
        "gpu_backend": jax.default_backend() == "gpu",
        "mode_probabilities": all(probability_checks.values()),
        "reset_state_finite": bool(np.isfinite(np.asarray(state.data.qpos)).all()),
    }

    return {
        "backend": jax.default_backend(),
        "brax_version": version("brax"),
        "checks": checks,
        "command_limits": {
            "x": list(x_limits),
            "y": list(y_limits),
            "yaw": list(yaw_limits),
        },
        "devices": [str(device) for device in jax.devices()],
        "expected_probabilities": expected_probabilities,
        "experiment": "g1_structured_command_sampling_smoke",
        "jax_version": jax.__version__,
        "mode_counts": mode_counts,
        "mode_probability_checks": probability_checks,
        "motionforge_revision": git_output(
            motionforge_root,
            "rev-parse",
            "HEAD",
        ),
        "motionforge_status_short": git_output(
            motionforge_root,
            "status",
            "--short",
        ).splitlines(),
        "mujoco_version": version("mujoco"),
        "observation_shape": list(state.obs["state"].shape),
        "observed_probabilities": observed_probabilities,
        "passed": all(checks.values()),
        "playground_revision": git_output(
            playground_root,
            "rev-parse",
            "HEAD",
        ),
        "playground_status_short": git_output(
            playground_root,
            "status",
            "--short",
        ).splitlines(),
        "playground_version": version("playground"),
        "probability_tolerances": probability_tolerances,
        "python_version": platform.python_version(),
        "qpos_shape": list(state.data.qpos.shape),
        "samples": config.samples,
        "seed": config.seed,
    }


def main(config: Config) -> None:
    result = run_smoke_test(config)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Config))
