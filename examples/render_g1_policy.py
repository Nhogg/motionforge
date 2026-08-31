from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapy
import mujoco
import numpy as np
import tyro
from mujoco_playground._src.locomotion.g1 import g1_constants
from mujoco_playground._src.locomotion.g1.base import get_assets
from reproduce_g1_policy import FixedCommandController


@dataclass(frozen=True)
class Config:
    duration: float = 10.0
    command_x: float = 0.5
    command_y: float = 0.0
    command_yaw: float = 0.0
    fps: int = 30
    width: int = 640
    height: int = 480
    output: Path = Path("logs/p1/g1_policy.mp4")


def main(config: Config) -> None:
    simulation_timestep = 0.002
    control_timestep = 0.02

    policy_path = (
        Path("/home/nhogg/mujoco_playground")
        / "mujoco_playground"
        / "experimental"
        / "sim2sim"
        / "onnx"
        / "g1_policy.onnx"
    )

    model = mujoco.MjModel.from_xml_path(
        g1_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        assets=get_assets(),
    )
    data = mujoco.MjData(model)

    keyframe = model.keyframe("knees_bent")
    mujoco.mj_resetDataKeyframe(model, data, keyframe.id)

    model.opt.timestep = simulation_timestep

    default_angles = np.array(keyframe.qpos[7:])
    data.ctrl[:] = default_angles
    mujoco.mj_forward(model, data)

    controller = FixedCommandController(
        policy_path=policy_path,
        default_angles=default_angles,
        command=np.array(
            [
                config.command_x,
                config.command_y,
                config.command_yaw,
            ],
            dtype=np.float32,
        ),
        control_timestep=control_timestep,
        simulation_timestep=simulation_timestep,
    )

    renderer = mujoco.Renderer(
        model,
        height=config.height,
        width=config.width,
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("torso_link").id
    camera.distance = 3.0
    camera.azimuth = 135.0
    camera.elevation = -15.0

    frames: list[np.ndarray] = []
    next_frame_time = 0.0

    while data.time < config.duration:
        controller.update(model, data)
        mujoco.mj_step(model, data)

        if data.time >= next_frame_time:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
            next_frame_time += 1.0 / config.fps

    renderer.close()

    config.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(
        config.output,
        frames,
        fps=config.fps,
    )

    print(
        {
            "frames": len(frames),
            "output": str(config.output),
            "simulation_time": float(data.time),
        }
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
