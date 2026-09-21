"""Render representative episodes from the P6 terrain curriculum.

The script uses the accepted G1 checkpoint, ordinary environment-sampled
commands, and the seeded terrain schedule used by the dataset generator. It
writes an MP4 plus a JSON sidecar describing every rendered episode.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import mediapy
import mujoco
import numpy as np
from mujoco import mjx

from motionforge.cli import run_hydra
from motionforge.controllers import G1LocomotionController
from motionforge.datasets import TerrainCurriculum, json_default
from motionforge.envs.g1_standing import G1StandingJoystick, default_config


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    episode_duration: float = 3.0
    rough_probabilities: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0)
    fps: int = 30
    width: int = 960
    height: int = 720
    naconmax: int = 16
    njmax: int = 128
    output: Path = Path("logs/p6/videos/terrain_curriculum_seed0.mp4")


def validate_config(config: Config) -> None:
    if config.episode_duration <= 0.0:
        raise ValueError("episode_duration must be positive")
    if config.fps <= 0:
        raise ValueError("fps must be positive")
    if config.width <= 0 or config.height <= 0:
        raise ValueError("video dimensions must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("contact capacities must be positive")


def main(config: Config) -> None:
    validate_config(config)
    curriculum = TerrainCurriculum(config.rough_probabilities, config.seed)
    environments: dict[str, G1StandingJoystick] = {}
    for name, task in (("flat", "flat_terrain"), ("rough", "rough_terrain")):
        environment_config = default_config()
        environment_config.impl = "warp"
        environment_config.naconmax = config.naconmax
        environment_config.njmax = config.njmax
        environment_config.push_config.enable = False
        environments[name] = G1StandingJoystick(
            config=environment_config,
            task=task,
        )

    controller = G1LocomotionController.from_checkpoint(
        config.checkpoint,
        environments["flat"],
        deterministic=True,
    )
    act = jax.jit(controller.act)
    resets = {
        name: jax.jit(environment.reset) for name, environment in environments.items()
    }
    steps = {
        name: jax.jit(environment.step) for name, environment in environments.items()
    }

    episode_count = len(config.rough_probabilities)
    schedule = tuple(
        curriculum.assignment(episode, episode / episode_count)
        for episode in range(episode_count)
    )
    frames: list[np.ndarray] = []
    episode_results: list[dict] = []

    for episode, (stage, probability, terrain) in enumerate(schedule):
        environment = environments[terrain]
        model = environment.mj_model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)
        renderer = mujoco.Renderer(
            model,
            width=config.width,
            height=config.height,
        )
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = model.body("torso_link").id
        camera.distance = 3.5
        camera.azimuth = 135.0
        camera.elevation = -20.0

        episode_seed = config.seed + episode
        rng = jax.random.PRNGKey(episode_seed)
        rng, reset_rng = jax.random.split(rng)
        state = resets[terrain](reset_rng)
        command = np.asarray(state.info["command"], dtype=float).tolist()
        control_timestep = float(environment.dt)
        rollout_steps = round(config.episode_duration / control_timestep)
        next_frame_time = 0.0
        episode_frames = 0
        terminated_at: float | None = None

        try:
            for step in range(rollout_steps):
                rng, action_rng = jax.random.split(rng)
                control = act(state.obs, action_rng)
                state = steps[terrain](state, control.normalized_action)
                simulation_time = (step + 1) * control_timestep

                if bool(np.asarray(state.done)):
                    terminated_at = simulation_time

                if simulation_time >= next_frame_time:
                    host_data = mjx.get_data(model, state.data)
                    renderer.update_scene(host_data, camera=camera)
                    frames.append(renderer.render().copy())
                    episode_frames += 1
                    next_frame_time += 1.0 / config.fps

                if terminated_at is not None:
                    break
        finally:
            renderer.close()

        episode_results.append(
            {
                "command": command,
                "episode": episode,
                "episode_frames": episode_frames,
                "episode_seed": episode_seed,
                "rough_probability": probability,
                "stage": stage,
                "terminated_at": terminated_at,
                "terrain": terrain,
            }
        )

    if not frames:
        raise RuntimeError("No video frames were captured")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(config.output, frames, fps=config.fps)
    result = {
        "backend": jax.default_backend(),
        "checkpoint": str(controller.checkpoint_path),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "episodes": episode_results,
        "experiment": "p6_terrain_curriculum_video",
        "frames": len(frames),
        "jax_version": jax.__version__,
        "mujoco_version": version("mujoco"),
        "output": str(config.output.resolve()),
        "python_version": platform.python_version(),
        "video_duration": len(frames) / config.fps,
    }
    sidecar = config.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(result, default=json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, default=json_default, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
