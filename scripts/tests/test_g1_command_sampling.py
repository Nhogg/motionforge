"""test_g1_command_sampling.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
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
    samples: int = 4096
    standing_probability: float = 0.30
    playground_root: Path = Path("../mujoco_playground")
    output: Path = Path("logs/p2/g1_command_sampling_a.json")


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
    if not 0.0 < config.standing_probability < 1.0:
        raise ValueError("--standing-probability must be between zero and one")

    environment = G1StandingJoystick(
        config_overrides={
            "standing_probability": config.standing_probability,
        }
    )

    reset_key = jax.random.PRNGKey(config.seed)
    sampling_key = jax.random.PRNGKey(config.seed + 1)

    state = environment.reset(reset_key)

    keys = jax.random.split(sampling_key, config.samples)
    sampled_commands = jax.vmap(environment.sample_command)(keys)
    commands = np.asarray(jax.device_get(sampled_commands))

    standing_mask = np.all(commands == 0.0, axis=1)
    moving_commands = commands[~standing_mask]

    standing_count = int(standing_mask.sum())
    moving_count = int((~standing_mask).sum())
    observed_probability = standing_count / config.samples

    expected_standard_error = math.sqrt(
        config.standing_probability
        * (1.0 - config.standing_probability)
        / config.samples
    )
    probability_tolerance = max(
        0.02,
        5.0 * expected_standard_error,
    )

    x_limits = environment._config.lin_vel_x
    y_limits = environment._config.lin_vel_y
    yaw_limits = environment._config.ang_vel_yaw

    moving_in_range = bool(
        moving_count > 0
        and np.all(moving_commands[:, 0] >= x_limits[0])
        and np.all(moving_commands[:, 0] <= x_limits[1])
        and np.all(moving_commands[:, 1] >= y_limits[0])
        and np.all(moving_commands[:, 1] <= y_limits[1])
        and np.all(moving_commands[:, 2] >= yaw_limits[0])
        and np.all(moving_commands[:, 2] <= yaw_limits[1])
    )

    motionforge_root = Path(__file__).resolve().parents[2]
    playground_root = config.playground_root.resolve()

    checks = {
        "commands_finite": bool(np.isfinite(commands).all()),
        "gpu_backend": jax.default_backend() == "gpu",
        "moving_commands_observed": moving_count > 0,
        "moving_commands_in_range": moving_in_range,
        "reset_state_finite": bool(np.isfinite(np.asarray(state.data.qpos)).all()),
        "standing_commands_observed": standing_count > 0,
        "standing_probability": (
            abs(observed_probability - config.standing_probability)
            <= probability_tolerance
        ),
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
        "experiment": "g1_standing_command_sampling_smoke",
        "expected_standing_probability": (config.standing_probability),
        "jax_version": jax.__version__,
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
        "moving_count": moving_count,
        "mujoco_version": version("mujoco"),
        "observed_standing_probability": observed_probability,
        "observation_shape": list(state.obs["state"].shape),
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
        "probability_standard_error": expected_standard_error,
        "probability_tolerance": probability_tolerance,
        "python_version": platform.python_version(),
        "qpos_shape": list(state.data.qpos.shape),
        "samples": config.samples,
        "seed": config.seed,
        "standing_count": standing_count,
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
