"""render_g1_checkpoint.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Render a trained G1 checkpoint executing a fixed command.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
import tyro
from brax.training import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx

from motionforge.compat.brax_checkpoint import load_ppo_network
from motionforge.envs.g1_standing import G1StandingJoystick, default_config

COMMAND_OBSERVATION_SLICE = slice(9, 12)


@dataclass(frozen=True)
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_standing_full_seed0/checkpoints/000191692800"
    )
    seed: int = 0

    command_x: float = 0.5
    command_y: float = 0.0
    command_yaw: float = 0.0
    stop_at: float | None = 5.0

    duration: float = 10.0
    fps: int = 30
    width: int = 960
    height: int = 720

    naconmax: int = 16
    njmax: int = 128

    output: Path = Path("logs/p2/videos/g1_checkpoint_best_forward_stop.mp4")


def force_command(state, command: jax.Array):
    """Put a command into both environment state and policy observation."""
    info = dict(state.info)
    info["command"] = command

    observation = dict(state.obs)
    observation["state"] = (
        observation["state"].at[COMMAND_OBSERVATION_SLICE].set(command)
    )
    observation["privileged_state"] = (
        observation["privileged_state"].at[COMMAND_OBSERVATION_SLICE].set(command)
    )

    return state.replace(info=info, obs=observation)


def validate_config(config: Config) -> None:
    if config.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if config.fps <= 0:
        raise ValueError("--fps must be positive")
    if config.width <= 0 or config.height <= 0:
        raise ValueError("--width and --height must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("--naconmax and --njmax must be positive")
    if config.stop_at is not None:
        if not 0.0 < config.stop_at < config.duration:
            raise ValueError("--stop-at must lie within the rollout duration")


def main(config: Config) -> None:
    validate_config(config)

    checkpoint_path = config.checkpoint.resolve()
    network_config_path = checkpoint_path / "ppo_network_config.json"

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    ppo_network = load_ppo_network(network_config_path)
    parameters = checkpoint.load(checkpoint_path)
    policy = ppo_networks.make_inference_fn(ppo_network)(
        parameters,
        deterministic=True,
    )

    environment_config = default_config()
    environment_config.impl = "warp"
    environment_config.naconmax = config.naconmax
    environment_config.njmax = config.njmax
    environment_config.push_config.enable = False

    environment = G1StandingJoystick(config=environment_config)

    control_timestep = float(environment.dt)
    rollout_steps = round(config.duration / control_timestep)

    moving_command = jp.array(
        [
            config.command_x,
            config.command_y,
            config.command_yaw,
        ],
        dtype=jp.float32,
    )
    stopped_command = jp.zeros(3, dtype=jp.float32)

    @jax.jit
    def controlled_step(state, command, action_rng):
        commanded_state = force_command(state, command)
        action, _ = policy(commanded_state.obs, action_rng)
        return environment.step(commanded_state, action)

    rng = jax.random.PRNGKey(config.seed)
    rng, reset_rng = jax.random.split(rng)
    state = environment.reset(reset_rng)

    model = environment.mj_model
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
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("torso_link").id
    camera.distance = 3.0
    camera.azimuth = 135.0
    camera.elevation = -15.0

    frames: list[np.ndarray] = []
    next_frame_time = 0.0
    terminated_at: float | None = None
    minimum_root_height = float("inf")

    try:
        for step in range(rollout_steps):
            simulation_time = step * control_timestep

            stopped = config.stop_at is not None and simulation_time >= config.stop_at
            command = stopped_command if stopped else moving_command

            rng, action_rng = jax.random.split(rng)
            state = controlled_step(state, command, action_rng)

            root_height = float(np.asarray(state.data.qpos[2]))
            minimum_root_height = min(
                minimum_root_height,
                root_height,
            )

            if terminated_at is None and bool(np.asarray(state.done)):
                terminated_at = simulation_time + control_timestep

            frame_time = simulation_time + control_timestep
            if frame_time >= next_frame_time:
                host_data = mjx.get_data(model, state.data)
                renderer.update_scene(host_data, camera=camera)
                frames.append(renderer.render().copy())
                next_frame_time += 1.0 / config.fps
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("No video frames were captured")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(
        config.output,
        frames,
        fps=config.fps,
    )

    sidecar_path = config.output.with_suffix(".json")
    result = {
        "backend": jax.default_backend(),
        "checkpoint": str(checkpoint_path),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "control_timestep": control_timestep,
        "experiment": "g1_checkpoint_video",
        "frames": len(frames),
        "jax_version": jax.__version__,
        "minimum_root_height": minimum_root_height,
        "mujoco_version": version("mujoco"),
        "output": str(config.output.resolve()),
        "python_version": platform.python_version(),
        "simulation_steps": rollout_steps,
        "terminated_at": terminated_at,
        "video_duration": len(frames) / config.fps,
    }

    sidecar_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Config))
