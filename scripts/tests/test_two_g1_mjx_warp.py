"""test_two_g1_mjx_warp.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
from mujoco import mjx

from motionforge.cli import run_hydra
from motionforge.envs.two_g1 import (
    build_two_g1_model,
    make_two_g1_data,
)


@dataclass
class Config:
    seed: int = 0
    steps: int = 10
    timestep: float = 0.002
    separation: float = 2.0
    naconmax: int = 32
    njmax: int = 256
    output: Path = Path("logs/p3/two_g1_mjx_warp_a.json")


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main(config: Config) -> None:
    if config.steps <= 0:
        raise ValueError("--steps must be positive")
    if config.naconmax <= 0:
        raise ValueError("--naconmax must be positive")
    if config.njmax <= 0:
        raise ValueError("--njmax must be positive")

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError("The two-G1 Warp test requires a GPU")

    device = gpu_devices[0]
    bundle = build_two_g1_model(timestep=config.timestep)
    cpu_data = make_two_g1_data(
        bundle,
        separation=config.separation,
    )

    warp_model = mjx.put_model(
        bundle.model,
        device=device,
        impl="warp",
    )
    initial_data = mjx.put_data(
        bundle.model,
        cpu_data,
        device=device,
        impl="warp",
        naconmax=config.naconmax,
        njmax=config.njmax,
    )

    def step_once(data, _):
        next_data = mjx.step(warp_model, data)
        return next_data, next_data._impl.nacon

    def rollout(data):
        return jax.lax.scan(
            step_once,
            data,
            xs=None,
            length=config.steps,
        )

    final_data, contact_counts = jax.jit(rollout)(initial_data)
    final_data.qpos.block_until_ready()

    final_qpos = np.asarray(final_data.qpos)
    final_qvel = np.asarray(final_data.qvel)
    final_ctrl = np.asarray(final_data.ctrl)

    root_heights = [
        float(final_qpos[agent.qpos_slice.start + 2]) for agent in bundle.agents
    ]
    maximum_contacts = int(np.asarray(contact_counts).max())

    expected_time = config.steps * config.timestep
    simulation_time = float(np.asarray(final_data.time))
    time_tolerance = config.timestep * 0.01

    checks = {
        "action_partition": (
            final_ctrl.shape == (58,)
            and len(bundle.agents[0].actuator_ids) == 29
            and len(bundle.agents[1].actuator_ids) == 29
        ),
        "contacts_present": maximum_contacts > 0,
        "dimensions": (final_qpos.shape == (72,) and final_qvel.shape == (70,)),
        "finite_state": bool(
            np.isfinite(final_qpos).all()
            and np.isfinite(final_qvel).all()
            and np.isfinite(final_ctrl).all()
        ),
        "gpu_backend": jax.default_backend() == "gpu",
        "robots_remain_above_floor": all(height > 0.7 for height in root_heights),
        "simulation_time": math.isclose(
            simulation_time,
            expected_time,
            rel_tol=0.0,
            abs_tol=time_tolerance,
        ),
    }

    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "device": str(device),
        "experiment": "two_g1_mjx_warp_smoke",
        "final_qpos_sha256": array_digest(final_qpos),
        "final_qvel_sha256": array_digest(final_qvel),
        "jax_version": jax.__version__,
        "maximum_contacts": maximum_contacts,
        "model": {
            "nq": bundle.model.nq,
            "nu": bundle.model.nu,
            "nv": bundle.model.nv,
        },
        "mujoco_mjx_version": version("mujoco-mjx"),
        "mujoco_version": version("mujoco"),
        "naconmax": config.naconmax,
        "njmax": config.njmax,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "root_heights": root_heights,
        "seed": config.seed,
        "simulation_time": simulation_time,
        "warp_version": version("warp-lang"),
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
