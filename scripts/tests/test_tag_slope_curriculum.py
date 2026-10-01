"""Verify per-reset flat and signed-slope sampling in MJX-Warp.

The test resets one terrain-curriculum environment across explicit seeds,
recovers the kinematic floor angle from MJX data, and checks reproducibility,
coverage, terrain-relative spawn clearance, and finite state. It writes the
sampled schedule and checks to JSON.
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import TagEnvironmentConfig, TwoG1TagEnvironment


@dataclass
class Config:
    seed: int = 0
    seeds: int = 12
    maximum_slope_degrees: float = 5.0
    separation: float = 2.0
    output: Path = Path("logs/p9/tag_slope_curriculum_a.json")


def _slope_degrees(quaternion: np.ndarray) -> float:
    return math.degrees(2.0 * math.atan2(quaternion[2], quaternion[0]))


def main(config: Config) -> None:
    if config.seeds <= 0:
        raise ValueError("seeds must be positive")
    if config.maximum_slope_degrees <= 0.0:
        raise ValueError("maximum_slope_degrees must be positive")

    environment = TwoG1TagEnvironment(
        TagEnvironmentConfig(
            separation=config.separation,
            slope_curriculum_degrees=config.maximum_slope_degrees,
        )
    )
    reset = jax.jit(environment.reset)
    records = []
    for offset in range(config.seeds):
        rollout_seed = config.seed + offset
        state = reset(jax.random.PRNGKey(rollout_seed))
        state.data.qpos.block_until_ready()
        quaternion = np.asarray(state.data.mocap_quat)[
            environment.model_bundle.terrain_mocap_id
        ]
        slope = _slope_degrees(quaternion)
        qpos = np.asarray(state.data.qpos)
        clearances = []
        for agent in environment.model_bundle.agents:
            root = qpos[agent.qpos_slice.start : agent.qpos_slice.start + 3]
            terrain_height = -math.tan(math.radians(slope)) * root[0]
            clearances.append(float(root[2] - terrain_height))
        records.append(
            {
                "finite": bool(np.isfinite(qpos).all()),
                "root_clearances": clearances,
                "seed": rollout_seed,
                "slope_degrees": slope,
            }
        )

    repeated = reset(jax.random.PRNGKey(config.seed))
    repeated.data.qpos.block_until_ready()
    repeated_quaternion = np.asarray(repeated.data.mocap_quat)[
        environment.model_bundle.terrain_mocap_id
    ]
    expected_slopes = {
        -config.maximum_slope_degrees,
        0.0,
        config.maximum_slope_degrees,
    }
    observed_slopes = {round(record["slope_degrees"], 5) for record in records}
    checks = {
        "all_choices_observed": observed_slopes == expected_slopes,
        "all_states_finite": all(record["finite"] for record in records),
        "choices_valid": all(
            round(record["slope_degrees"], 5) in expected_slopes
            for record in records
        ),
        "deterministic_first_reset": bool(
            np.allclose(
                repeated_quaternion,
                np.asarray(
                    [
                        math.cos(
                            math.radians(records[0]["slope_degrees"]) / 2.0
                        ),
                        0.0,
                        math.sin(
                            math.radians(records[0]["slope_degrees"]) / 2.0
                        ),
                        0.0,
                    ]
                ),
                atol=1e-7,
            )
        ),
        "equal_terrain_clearance": all(
            np.allclose(record["root_clearances"], 0.755, atol=1e-6)
            for record in records
        ),
        "gpu_backend": jax.default_backend() == "gpu",
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "p9_tag_slope_curriculum_reset",
        "jax_version": jax.__version__,
        "mujoco_version": version("mujoco"),
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "records": records,
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
