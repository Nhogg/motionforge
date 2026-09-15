"""Verify the P3 tag arena uses one level, shared flat floor."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import build_two_g1_model, make_two_g1_data


@dataclass
class Config:
    seed: int = 0
    separation: float = 2.0
    timestep: float = 0.002
    output: Path = Path("logs/p3/tag_flat_terrain_a.json")


def main(config: Config) -> None:
    bundle = build_two_g1_model(timestep=config.timestep)
    model = bundle.model
    data = make_two_g1_data(bundle, separation=config.separation)

    floor_id = model.geom("floor").id
    floor_position = model.geom_pos[floor_id].copy()
    floor_quaternion = model.geom_quat[floor_id].copy()
    plane_ids = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    root_heights = np.asarray(
        [data.qpos[agent.qpos_slice.start + 2] for agent in bundle.agents]
    )

    checks = {
        "agents_above_floor": bool(np.all(root_heights > floor_position[2])),
        "floor_is_level": bool(
            np.allclose(floor_quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        ),
        "floor_is_plane": bool(
            model.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_PLANE
        ),
        "floor_named": bool(model.geom(floor_id).name == "floor"),
        "floor_on_world_zero": bool(
            np.allclose(floor_position, [0.0, 0.0, 0.0], atol=1e-12)
        ),
        "no_heightfields": bool(model.nhfield == 0),
        "one_shared_plane": bool(
            len(plane_ids) == 1 and int(plane_ids[0]) == floor_id
        ),
        "root_heights_equal": bool(
            np.isclose(root_heights[0], root_heights[1], atol=1e-12)
        ),
        "timestep_preserved": bool(
            np.isclose(model.opt.timestep, config.timestep, atol=1e-12)
        ),
    }

    result = {
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_flat_terrain",
        "floor": {
            "geom_id": floor_id,
            "position": floor_position.tolist(),
            "quaternion": floor_quaternion.tolist(),
            "type": mujoco.mjtGeom(model.geom_type[floor_id]).name,
        },
        "heightfield_count": model.nhfield,
        "mujoco_version": version("mujoco"),
        "passed": bool(all(checks.values())),
        "plane_geom_ids": plane_ids.tolist(),
        "python_version": platform.python_version(),
        "root_heights": root_heights.tolist(),
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
