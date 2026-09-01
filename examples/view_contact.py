from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapy
import mujoco
import numpy as np
import tyro
from dm_control.locomotion import soccer


@dataclass(frozen=True)
class Config:
    seed: int = 0
    duration: float = 0.75
    initial_separation: float = 0.8
    impact_speed: float = 1.5
    capture_fps: int = 120
    playback_fps: int = 30
    width: int = 960
    height: int = 720
    output: Path = Path("logs/p1/humanoid_contact.mp4")


def main(config: Config) -> None:
    environment = soccer.load(
        team_size=1,
        time_limit=max(config.duration, 1.0),
        random_state=np.random.RandomState(config.seed),
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=soccer.WalkerType.HUMANOID,
    )
    environment.reset()

    model = environment.physics.model.ptr
    data = environment.physics.data.ptr

    home_joint = model.joint("home0/")
    away_joint = model.joint("away0/")

    home_qpos = int(home_joint.qposadr[0])
    away_qpos = int(away_joint.qposadr[0])
    home_dof = int(home_joint.dofadr[0])
    away_dof = int(away_joint.dofadr[0])

    data.qpos[home_qpos] = -config.initial_separation / 2.0
    data.qpos[home_qpos + 1] = 0.0

    data.qpos[away_qpos] = config.initial_separation / 2.0
    data.qpos[away_qpos + 1] = 0.0

    data.qvel[:] = 0.0
    data.qvel[home_dof] = config.impact_speed
    data.qvel[away_dof] = -config.impact_speed
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)

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
        height=config.height,
        width=config.width,
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = 3.0
    camera.azimuth = 90.0
    camera.elevation = -10.0

    frames: list[np.ndarray] = []
    next_frame_time = 0.0

    while data.time < config.duration:
        mujoco.mj_step(model, data)

        if data.time >= next_frame_time:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
            next_frame_time += 1.0 / config.capture_fps

    renderer.close()
    environment.close()

    config.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(
        config.output,
        frames,
        fps=config.playback_fps,
    )

    print(
        {
            "frames": len(frames),
            "output": str(config.output),
            "playback_seconds": len(frames) / config.playback_fps,
        }
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
