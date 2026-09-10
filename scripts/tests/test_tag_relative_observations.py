"""test_tag_relative_observations.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs.tag_observations import (
    build_tag_observation_layout,
    relative_planar_observation,
)
from motionforge.envs.two_g1 import (
    build_two_g1_model,
    make_two_g1_data,
)


@dataclass
class Config:
    seed: int = 0
    separation: float = 2.0
    output: Path = Path("logs/p3/tag_relative_observations_a.json")


def main(config: Config) -> None:
    bundle = build_two_g1_model()
    data = make_two_g1_data(
        bundle,
        separation=config.separation,
    )
    observation_layout = build_tag_observation_layout(bundle)

    agent0 = bundle.agents[0]
    agent1 = bundle.agents[1]

    # Rotate agent 0 by +90 degrees about world Z.
    half_angle = np.pi / 4.0
    quaternion_start = agent0.qpos_slice.start + 3
    data.qpos[quaternion_start : quaternion_start + 4] = np.array(
        [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]
    )

    # Assign distinct world-frame planar root velocities.
    data.qvel[agent0.qvel_slice.start : agent0.qvel_slice.start + 2] = np.array(
        [0.2, -0.1]
    )
    data.qvel[agent1.qvel_slice.start : agent1.qvel_slice.start + 2] = np.array(
        [0.5, 0.3]
    )

    mujoco.mj_forward(bundle.model, data)

    observation0 = relative_planar_observation(
        data,
        observation_layout,
        observer_index=0,
    )
    observation1 = relative_planar_observation(
        data,
        observation_layout,
        observer_index=1,
    )

    position0 = np.asarray(observation0.position)
    velocity0 = np.asarray(observation0.velocity)
    position1 = np.asarray(observation1.position)
    velocity1 = np.asarray(observation1.velocity)

    # Agent 0 faces world +Y, so agent 1 is two metres forward.
    expected_position0 = np.array([config.separation, 0.0])
    expected_velocity0 = np.array([0.4, -0.3])

    # Agent 1 retains identity yaw and sees agent 0 at world -Y.
    expected_position1 = np.array([0.0, -config.separation])
    expected_velocity1 = np.array([-0.3, -0.4])

    checks = {
        "agent0_position": bool(
            np.allclose(
                position0,
                expected_position0,
                atol=1e-6,
            )
        ),
        "agent0_velocity": bool(
            np.allclose(
                velocity0,
                expected_velocity0,
                atol=1e-6,
            )
        ),
        "agent1_position": bool(
            np.allclose(
                position1,
                expected_position1,
                atol=1e-6,
            )
        ),
        "agent1_velocity": bool(
            np.allclose(
                velocity1,
                expected_velocity1,
                atol=1e-6,
            )
        ),
        "finite_values": bool(
            np.isfinite(position0).all()
            and np.isfinite(velocity0).all()
            and np.isfinite(position1).all()
            and np.isfinite(velocity1).all()
        ),
        "opposite_world_displacement": bool(
            np.isclose(
                np.linalg.norm(position0),
                np.linalg.norm(position1),
                atol=1e-6,
            )
        ),
        "observation_shapes": (
            position0.shape == (2,)
            and velocity0.shape == (2,)
            and position1.shape == (2,)
            and velocity1.shape == (2,)
        ),
    }

    result = {
        "agent_observations": [
            {
                "observer_index": 0,
                "relative_position": position0.tolist(),
                "relative_velocity": velocity0.tolist(),
            },
            {
                "observer_index": 1,
                "relative_position": position1.tolist(),
                "relative_velocity": velocity1.tolist(),
            },
        ],
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "tag_relative_observation_frames",
        "passed": all(checks.values()),
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
