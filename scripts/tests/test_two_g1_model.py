"""test_two_g1_model.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import tyro

from motionforge.envs.two_g1 import (
    build_two_g1_model,
    make_two_g1_data,
)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    steps: int = 10
    timestep: float = 0.002
    separation: float = 2.0
    output: Path = Path("logs/p3/two_g1_model_a.json")


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main(config: Config) -> None:
    if config.steps <= 0:
        raise ValueError("--steps must be positive")

    bundle = build_two_g1_model(timestep=config.timestep)
    data = make_two_g1_data(bundle, separation=config.separation)

    initial_positions = [
        data.qpos[agent.qpos_slice.start : agent.qpos_slice.start + 3].copy().tolist()
        for agent in bundle.agents
    ]

    for _ in range(config.steps):
        mujoco.mj_step(bundle.model, data)

    expected_time = config.steps * config.timestep
    final_positions = [
        data.qpos[agent.qpos_slice.start : agent.qpos_slice.start + 3].copy().tolist()
        for agent in bundle.agents
    ]

    checks = {
        "actuator_partition": (
            len(bundle.agents[0].actuator_ids) == 29
            and len(bundle.agents[1].actuator_ids) == 29
            and set(bundle.agents[0].actuator_ids).isdisjoint(
                bundle.agents[1].actuator_ids
            )
        ),
        "dimensions": (
            bundle.model.nq == 72 and bundle.model.nv == 70 and bundle.model.nu == 58
        ),
        "finite_state": bool(
            np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        ),
        "namespaced_roots": (
            bundle.model.body("agent0/pelvis").id >= 0
            and bundle.model.body("agent1/pelvis").id >= 0
        ),
        "separate_initial_positions": bool(
            np.isclose(
                initial_positions[1][1] - initial_positions[0][1],
                config.separation,
            )
        ),
        "simulation_time": bool(np.isclose(data.time, expected_time, atol=1e-12)),
    }

    result = {
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_model_smoke",
        "final_positions": final_positions,
        "final_qpos_sha256": array_digest(data.qpos),
        "final_qvel_sha256": array_digest(data.qvel),
        "initial_positions": initial_positions,
        "model": {
            "nbody": bundle.model.nbody,
            "nq": bundle.model.nq,
            "nu": bundle.model.nu,
            "nv": bundle.model.nv,
        },
        "mujoco_version": version("mujoco"),
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "seed": config.seed,
        "simulation_time": float(data.time),
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
    main(tyro.cli(Config))
