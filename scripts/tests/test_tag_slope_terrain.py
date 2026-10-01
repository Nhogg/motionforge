"""Smoke-test deterministic two-G1 tag resets and MJX-Warp steps on a slope.

The test builds one shared planar slope, verifies its geometry and reset
clearances, advances both robots with their accepted standing targets, and
writes finite-state and physical-survival evidence to JSON.
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
    slope_degrees: float = 5.0
    steps: int = 10
    separation: float = 2.0
    naconmax: int = 64
    njmax: int = 256
    output: Path = Path("logs/p9/tag_slope_terrain_a.json")


def main(config: Config) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.slope_degrees == 0.0:
        raise ValueError("slope_degrees must be nonzero for the slope test")

    environment = TwoG1TagEnvironment(
        TagEnvironmentConfig(
            separation=config.separation,
            slope_degrees=config.slope_degrees,
            naconmax=config.naconmax,
            njmax=config.njmax,
        )
    )
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    state = reset(jax.random.PRNGKey(config.seed))

    initial_qpos = np.asarray(state.data.qpos)
    slope_radians = math.radians(config.slope_degrees)
    expected_normal = np.asarray(
        [math.sin(slope_radians), 0.0, math.cos(slope_radians)]
    )
    floor_id = environment.model_bundle.model.geom("floor").id
    floor_rotation = np.asarray(state.data.geom_xmat[floor_id]).reshape(3, 3)
    floor_normal = floor_rotation[:, 2]
    initial_clearances = []
    for agent in environment.model_bundle.agents:
        root = initial_qpos[agent.qpos_slice.start : agent.qpos_slice.start + 3]
        terrain_height = -math.tan(slope_radians) * root[0]
        initial_clearances.append(float(root[2] - terrain_height))

    for _ in range(config.steps):
        state = step(state, environment.default_joint_targets)
        state.data.qpos.block_until_ready()

    final_qpos = np.asarray(state.data.qpos)
    final_qvel = np.asarray(state.data.qvel)
    checks = {
        "agents_begin_above_slope": all(value > 0.7 for value in initial_clearances),
        "agents_remain_upright": bool(
            np.all(np.asarray(state.diagnostics.up_alignment) > 0.9)
        ),
        "finite_state": bool(
            np.isfinite(final_qpos).all()
            and np.isfinite(final_qvel).all()
            and np.isfinite(np.asarray(state.observation.relative_position)).all()
        ),
        "gpu_backend": jax.default_backend() == "gpu",
        "no_early_termination": not bool(np.asarray(state.done)),
        "reset_clearances_match": bool(
            np.allclose(initial_clearances, initial_clearances[0], atol=1e-6)
        ),
        "slope_normal": bool(
            np.allclose(floor_normal, expected_normal, atol=1e-7)
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "p9_two_g1_slope_terrain_smoke",
        "floor_normal": floor_normal.tolist(),
        "initial_root_clearances": initial_clearances,
        "jax_version": jax.__version__,
        "mujoco_version": version("mujoco"),
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "seed": config.seed,
        "simulation_time": float(np.asarray(state.data.time)),
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
