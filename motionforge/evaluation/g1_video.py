"""g1_video.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Reusable deterministic video rendering for G1 policies.

    The policy and dynamics execute through MJX. Selected states are transferred
    to native MuJoCo for offscreen rendering and written to a local MP4.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
from mujoco import mjx

COMMAND_OBSERVATION_SLICE = slice(9, 12)


def force_command(state: Any, command: jax.Array) -> Any:
    """Write a command into environment state and policy observations."""
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


def render_g1_policy_video(
    *,
    environment: Any,
    make_policy: Callable[..., Any],
    parameters: Any,
    output: Path,
    seed: int,
    command: tuple[float, float, float],
    duration: float,
    fps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Render a deterministic fixed-command G1 rollout to MP4."""
    if duration <= 0.0:
        raise ValueError("Video duration must be positive")
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("Video dimensions must be positive")

    policy = make_policy(parameters, deterministic=True)
    command_array = jp.asarray(command, dtype=jp.float32)

    @jax.jit
    def controlled_step(state, action_rng):
        commanded_state = force_command(state, command_array)
        action, _ = policy(commanded_state.obs, action_rng)
        return environment.step(commanded_state, action)

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = environment.reset(reset_rng)

    control_timestep = float(environment.dt)
    rollout_steps = round(duration / control_timestep)

    model = environment.mj_model
    model.vis.global_.offwidth = max(
        model.vis.global_.offwidth,
        width,
    )
    model.vis.global_.offheight = max(
        model.vis.global_.offheight,
        height,
    )

    renderer = mujoco.Renderer(
        model,
        width=width,
        height=height,
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("torso_link").id
    camera.distance = 3.0
    camera.azimuth = 135.0
    camera.elevation = -15.0

    frames: list[np.ndarray] = []
    next_frame_time = 0.0
    minimum_root_height = float("inf")
    terminated_at: float | None = None

    try:
        for step in range(rollout_steps):
            rng, action_rng = jax.random.split(rng)
            state = controlled_step(state, action_rng)

            simulation_time = (step + 1) * control_timestep
            root_height = float(np.asarray(state.data.qpos[2]))
            minimum_root_height = min(
                minimum_root_height,
                root_height,
            )

            if terminated_at is None and bool(np.asarray(state.done)):
                terminated_at = simulation_time

            if simulation_time >= next_frame_time:
                host_data = mjx.get_data(model, state.data)
                renderer.update_scene(host_data, camera=camera)
                frames.append(renderer.render().copy())
                next_frame_time += 1.0 / fps
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("No video frames were captured")

    output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(output, frames, fps=fps)

    return {
        "command": list(command),
        "control_timestep": control_timestep,
        "duration": duration,
        "fps": fps,
        "frames": len(frames),
        "height": height,
        "minimum_root_height": minimum_root_height,
        "output": str(output.resolve()),
        "seed": seed,
        "simulation_steps": rollout_steps,
        "terminated_at": terminated_at,
        "width": width,
    }
