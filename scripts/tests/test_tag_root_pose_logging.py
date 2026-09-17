"""Record and validate per-timestep root poses from the tag environment.

The seeded smoke experiment writes append-only JSONL trajectory records and a
JSON summary. It uses default joint targets so it tests logging independently
of a learned controller.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import mujoco
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import TagEnvironmentConfig, TwoG1TagEnvironment
from motionforge.logging import (
    TAG_TRAJECTORY_SCHEMA_VERSION,
    build_tag_root_pose_layout,
    extract_tag_root_pose,
)


@dataclass
class Config:
    seed: int = 0
    steps: int = 10
    separation: float = 2.0
    naconmax: int = 32
    njmax: int = 256
    output: Path = Path("logs/p5/tag_root_pose_a.jsonl")
    summary: Path = Path("logs/p5/tag_root_pose_a_summary.json")


def _validate_config(config: Config) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.separation <= 0.0:
        raise ValueError("separation must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("contact capacities must be positive")


def main(config: Config) -> None:
    _validate_config(config)
    environment = TwoG1TagEnvironment(
        TagEnvironmentConfig(
            separation=config.separation,
            naconmax=config.naconmax,
            njmax=config.njmax,
        )
    )
    layout = build_tag_root_pose_layout(environment.model_bundle)
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    extract = jax.jit(lambda state: extract_tag_root_pose(state.data, layout))

    state = reset(jax.random.PRNGKey(config.seed))
    records = []
    for timestep in range(config.steps + 1):
        root_pose = extract(state)
        root_pose.position.block_until_ready()
        position = np.asarray(root_pose.position)
        orientation = np.asarray(root_pose.orientation_wxyz)
        records.append(
            {
                "episode_seed": config.seed,
                "orientation_wxyz": orientation.tolist(),
                "position": position.tolist(),
                "schema_version": TAG_TRAJECTORY_SCHEMA_VERSION,
                "simulation_time": float(np.asarray(state.diagnostics.elapsed_seconds)),
                "timestep": timestep,
            }
        )
        if timestep < config.steps:
            state = step(state, environment.default_joint_targets)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    positions = np.asarray([record["position"] for record in records])
    orientations = np.asarray([record["orientation_wxyz"] for record in records])
    quaternion_norms = np.linalg.norm(orientations, axis=-1)
    checks = {
        "backend_gpu": jax.default_backend() == "gpu",
        "finite_values": bool(
            np.isfinite(positions).all() and np.isfinite(orientations).all()
        ),
        "orientation_shape": orientations.shape == (config.steps + 1, 2, 4),
        "positions_distinct": bool(
            np.linalg.norm(positions[0, 1] - positions[0, 0]) > 0.0
        ),
        "position_shape": positions.shape == (config.steps + 1, 2, 3),
        "quaternions_normalized": bool(np.allclose(quaternion_norms, 1.0, atol=1e-5)),
        "record_count": len(records) == config.steps + 1,
        "simulation_time": bool(
            np.isclose(
                records[-1]["simulation_time"],
                config.steps * environment.config.control_timestep,
            )
        ),
        "timesteps_contiguous": [record["timestep"] for record in records]
        == list(range(config.steps + 1)),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
            "summary": str(config.summary),
        },
        "experiment": "tag_root_pose_logging",
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "record_count": len(records),
        "schema_version": TAG_TRAJECTORY_SCHEMA_VERSION,
        "seed": config.seed,
    }
    config.summary.parent.mkdir(parents=True, exist_ok=True)
    config.summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
