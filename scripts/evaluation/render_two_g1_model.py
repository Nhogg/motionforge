"""render_two_g1_model.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Render the initialized two-G1 flat-ground model as an Mp4.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import mediapy
import mujoco
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs.two_g1 import (
    build_two_g1_model,
    make_two_g1_data,
)


@dataclass
class Config:
    separation: float = 2.0
    duration: float = 5.0
    fps: int = 30
    width: int = 960
    height: int = 720
    camera_distance: float = 4.0
    orbit_degrees: float = 180.0
    output: Path = Path("logs/p3/videos/two_g1_environment.mp4")


def validate_config(config: Config) -> None:
    if config.separation <= 0.0:
        raise ValueError("--separation must be positive")
    if config.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if config.fps <= 0:
        raise ValueError("--fps must be positive")
    if config.width <= 0 or config.height <= 0:
        raise ValueError("--width and --height must be positive")
    if config.camera_distance <= 0.0:
        raise ValueError("--camera-distance must be positive")


def main(config: Config) -> None:
    validate_config(config)

    bundle = build_two_g1_model()
    data = make_two_g1_data(
        bundle,
        separation=config.separation,
    )

    model = bundle.model
    model.vis.global_.offwidth = max(
        model.vis.global_.offwidth,
        config.width,
    )
    model.vis.global_.offheight = max(
        model.vis.global_.offheight,
        config.height,
    )

    renderer = mujoco.Renderer(
        model,
        width=config.width,
        height=config.height,
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, 0.0, 0.7])
    camera.distance = config.camera_distance
    camera.elevation = -15.0

    frame_count = round(config.duration * config.fps)
    frames: list[np.ndarray] = []

    try:
        for frame_index in range(frame_count):
            fraction = frame_index / max(frame_count - 1, 1)
            camera.azimuth = 90.0 + fraction * config.orbit_degrees

            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("No frames were rendered")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(
        config.output,
        frames,
        fps=config.fps,
    )

    result = {
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_environment_render",
        "frames": len(frames),
        "model": {
            "nbody": model.nbody,
            "nq": model.nq,
            "nu": model.nu,
            "nv": model.nv,
        },
        "mujoco_version": version("mujoco"),
        "output": str(config.output.resolve()),
        "python_version": platform.python_version(),
        "root_positions": [
            data.qpos[agent.qpos_slice.start : agent.qpos_slice.start + 3].tolist()
            for agent in bundle.agents
        ],
        "video_duration": len(frames) / config.fps,
    }

    sidecar = config.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
