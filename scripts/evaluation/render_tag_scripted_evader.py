"""Render a two-G1 rollout driven by scripted tag policies.

Agent 1 always runs the boundary-aware scripted evader. Agent 0 can either
stand still for isolated evader diagnostics or run the scripted pursuer for a
complete tag rollout. Both agents use the same accepted locomotion checkpoint.
The script writes MP4 video and a JSON sidecar and requires MJX-Warp, a GPU
backend, and an OpenGL context.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
from mujoco import mjx

from motionforge.cli import run_hydra
from motionforge.controllers import (
    G1LocomotionController,
    build_g1_tag_policy_observation_layout,
    g1_tag_policy_observation,
)
from motionforge.envs import TagEnvironmentConfig, TwoG1TagEnvironment
from motionforge.envs.g1_standing import G1StandingJoystick, default_config
from motionforge.policies import (
    ScriptedEvaderConfig,
    ScriptedPursuerConfig,
    scripted_evader_command,
    scripted_pursuer_command,
)


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    separation: float = 2.0
    pursuer_active: bool = False
    duration: float = 10.0
    fps: int = 30
    width: int = 960
    height: int = 720
    camera_distance: float = 5.0
    naconmax: int = 64
    njmax: int = 256
    output: Path = Path("logs/p4/videos/tag_scripted_evader_a.mp4")


def _validate_config(config: Config) -> None:
    if config.duration <= 0.0:
        raise ValueError("duration must be positive")
    if config.fps <= 0:
        raise ValueError("fps must be positive")
    if config.width <= 0 or config.height <= 0:
        raise ValueError("width and height must be positive")
    if config.camera_distance <= 0.0:
        raise ValueError("camera_distance must be positive")


def main(config: Config) -> None:
    _validate_config(config)
    checkpoint_path = config.checkpoint.resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    source_config = default_config()
    source_config.impl = "warp"
    source_config.naconmax = 16
    source_config.njmax = 128
    source_config.push_config.enable = False
    source_environment = G1StandingJoystick(config=source_config)
    controller = G1LocomotionController.from_checkpoint(
        checkpoint_path,
        source_environment,
    )

    environment = TwoG1TagEnvironment(
        TagEnvironmentConfig(
            separation=config.separation,
            episode_duration=config.duration,
            naconmax=config.naconmax,
            njmax=config.njmax,
        )
    )
    observation_layout = build_g1_tag_policy_observation_layout(
        environment.model_bundle
    )
    evader_config = ScriptedEvaderConfig()
    pursuer_config = ScriptedPursuerConfig()
    default_pose = jp.asarray(source_environment._default_pose)
    action_scale = float(source_environment._config.action_scale)
    phase_dt = 2.0 * jp.pi * environment.config.control_timestep * 1.375

    @jax.jit
    def controlled_step(state, last_actions, phases, rng):
        evader_command = scripted_evader_command(
            state.observation.relative_position[1],
            state.observation.arena_center_position[1],
            evader_config,
        )
        pursuer_command = scripted_pursuer_command(
            state.observation.relative_position[0],
            pursuer_config,
        )
        pursuer_command = jp.where(
            config.pursuer_active,
            pursuer_command,
            jp.zeros(3),
        )
        commands = jp.stack([pursuer_command, evader_command])
        rng, agent0_rng, agent1_rng = jax.random.split(rng, 3)
        observation0 = g1_tag_policy_observation(
            state.data,
            observation_layout,
            0,
            commands[0],
            last_actions[0],
            phases[0],
            default_pose,
        )
        observation1 = g1_tag_policy_observation(
            state.data,
            observation_layout,
            1,
            commands[1],
            last_actions[1],
            phases[1],
            default_pose,
        )
        action0 = controller.act(observation0, agent0_rng).normalized_action
        action1 = controller.act(observation1, agent1_rng).normalized_action
        actions = jp.stack([action0, action1])
        joint_targets = default_pose + actions * action_scale
        next_state = environment.step(state, joint_targets)
        next_phases = jp.fmod(phases + phase_dt + jp.pi, 2.0 * jp.pi) - jp.pi
        return next_state, actions, next_phases, rng, commands

    rng = jax.random.PRNGKey(config.seed)
    rng, reset_rng = jax.random.split(rng)
    state = jax.jit(environment.reset)(reset_rng)
    last_actions = jp.zeros((2, 29))
    phases = jp.asarray([[0.0, jp.pi], [0.0, jp.pi]])

    model = environment.model_bundle.model
    for geom_id in environment.contact_layout.agent0_geom_ids:
        model.geom_rgba[geom_id, :3] = np.asarray([0.85, 0.25, 0.20])
    for geom_id in environment.contact_layout.agent1_geom_ids:
        model.geom_rgba[geom_id, :3] = np.asarray([0.20, 0.45, 0.90])
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)

    renderer = mujoco.Renderer(model, width=config.width, height=config.height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = config.camera_distance
    camera.azimuth = 135.0
    camera.elevation = -20.0

    frames: list[np.ndarray] = []
    next_frame_time = 0.0
    rollout_steps = round(config.duration / environment.config.control_timestep)
    commands = jp.zeros((2, 3))
    terminated_at = None
    minimum_agent_distance = float("inf")

    try:
        for step_index in range(rollout_steps):
            state, last_actions, phases, rng, commands = controlled_step(
                state,
                last_actions,
                phases,
                rng,
            )
            state.data.qpos.block_until_ready()
            frame_time = (step_index + 1) * environment.config.control_timestep

            roots = np.stack(
                [
                    np.asarray(
                        state.data.qpos[
                            agent.qpos_slice.start : agent.qpos_slice.start + 3
                        ]
                    )
                    for agent in environment.model_bundle.agents
                ]
            )
            minimum_agent_distance = min(
                minimum_agent_distance,
                float(np.linalg.norm(roots[1, :2] - roots[0, :2])),
            )
            if frame_time >= next_frame_time:
                camera.lookat[:] = roots.mean(axis=0)
                camera.lookat[2] = 0.7
                host_data = mjx.get_data(model, state.data)
                renderer.update_scene(host_data, camera=camera)
                frames.append(renderer.render().copy())
                next_frame_time += 1.0 / config.fps

            if bool(np.asarray(state.done)):
                terminated_at = frame_time
                break
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("No video frames were captured")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(config.output, frames, fps=config.fps)
    result = {
        "backend": jax.default_backend(),
        "checkpoint": str(checkpoint_path),
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "output": str(config.output),
        },
        "experiment": (
            "tag_scripted_complete_video"
            if config.pursuer_active
            else "tag_scripted_evader_video"
        ),
        "final_commands": np.asarray(commands).tolist(),
        "frames": len(frames),
        "minimum_root_height": float(
            min(np.asarray(state.diagnostics.root_height).tolist())
        ),
        "minimum_agent_distance": minimum_agent_distance,
        "mujoco_version": mujoco.__version__,
        "output": str(config.output.resolve()),
        "python_version": platform.python_version(),
        "terminated_at": terminated_at,
        "termination": {
            "fallen": np.asarray(state.termination.fallen).tolist(),
            "out_of_bounds": np.asarray(state.termination.out_of_bounds).tolist(),
            "tagged": bool(np.asarray(state.termination.tagged)),
            "timed_out": bool(np.asarray(state.termination.timed_out)),
        },
        "video_duration": len(frames) / config.fps,
    }
    config.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
