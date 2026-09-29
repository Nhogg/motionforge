"""Render one learned pursuer-versus-evader TAG matchup to MP4 and JSON.

Both high-level policies are deterministic frozen PPO checkpoints and share
the accepted P2 locomotion controller. Frames are captured at the high-level
control rate from the integrated MJX-Warp environment until termination.
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
from motionforge.envs import (
    TagEnvironmentConfig,
    TagPursuerConfig,
    TagPursuerEnvironment,
)
from motionforge.policies import FrozenLearnedPursuer


@dataclass
class Config:
    pursuer_checkpoint: Path = Path("???")
    evader_checkpoint: Path = Path("???")
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    pursuer_fixed_noise_std: float = 0.2
    evader_fixed_noise_std: float = 0.2
    seed: int = 1000
    duration: float = 20.0
    separation: float = 2.0
    action_repeat: int = 5
    width: int = 960
    height: int = 720
    camera_distance: float = 5.0
    naconmax: int = 64
    njmax: int = 256
    output: Path = Path("logs/p8/videos/tag_learned_matchup.mp4")


def main(config: Config) -> None:
    if config.duration <= 0.0 or config.action_repeat <= 0:
        raise ValueError("duration and action_repeat must be positive")
    environment = TagPursuerEnvironment(
        locomotion_checkpoint=config.locomotion_checkpoint,
        evader_checkpoint=config.evader_checkpoint,
        evader_fixed_noise_std=config.evader_fixed_noise_std,
        tag_config=TagEnvironmentConfig(
            episode_duration=config.duration,
            separation=config.separation,
            naconmax=config.naconmax,
            njmax=config.njmax,
        ),
        pursuer_config=TagPursuerConfig(action_repeat=config.action_repeat),
    )
    pursuer = FrozenLearnedPursuer(
        agent_index=environment.pursuer_index,
        checkpoint_path=config.pursuer_checkpoint,
        fixed_noise_std=config.pursuer_fixed_noise_std,
    )
    action_scale = jp.asarray([1.0, 0.5, 1.0])

    @jax.jit
    def policy_step(state):
        normalized_action = pursuer.command(state.obs) / action_scale
        return environment.step(state, normalized_action)

    state = jax.jit(environment.reset)(jax.random.PRNGKey(config.seed))
    model = environment.tag_environment.model_bundle.model
    for geom_id in environment.tag_environment.contact_layout.agent0_geom_ids:
        model.geom_rgba[geom_id, :3] = np.asarray([0.85, 0.25, 0.20])
    for geom_id in environment.tag_environment.contact_layout.agent1_geom_ids:
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
    terminated_at = None
    maximum_steps = round(config.duration / (0.02 * config.action_repeat))
    try:
        for step in range(maximum_steps):
            state = policy_step(state)
            state.pipeline_state.tag_state.data.qpos.block_until_ready()
            tag_state = state.pipeline_state.tag_state
            roots = np.stack(
                [
                    np.asarray(
                        tag_state.data.qpos[
                            agent.qpos_slice.start : agent.qpos_slice.start + 3
                        ]
                    )
                    for agent in environment.tag_environment.model_bundle.agents
                ]
            )
            camera.lookat[:] = roots.mean(axis=0)
            camera.lookat[2] = 0.7
            host_data = mjx.get_data(model, tag_state.data)
            renderer.update_scene(host_data, camera=camera)
            frames.append(renderer.render().copy())
            if bool(np.asarray(state.done)):
                terminated_at = (step + 1) * 0.02 * config.action_repeat
                break
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("no video frames captured")
    fps = round(1.0 / (0.02 * config.action_repeat))
    config.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(config.output, frames, fps=fps)
    termination = state.pipeline_state.tag_state.termination
    result = {
        "backend": jax.default_backend(),
        "config": {
            **asdict(config),
            "evader_checkpoint": str(config.evader_checkpoint),
            "locomotion_checkpoint": str(config.locomotion_checkpoint),
            "output": str(config.output),
            "pursuer_checkpoint": str(config.pursuer_checkpoint),
        },
        "experiment": "p8_tag_learned_matchup_video",
        "fps": fps,
        "frames": len(frames),
        "mujoco_version": mujoco.__version__,
        "output": str(config.output.resolve()),
        "python_version": platform.python_version(),
        "terminated_at": terminated_at,
        "termination": {
            "fallen": np.asarray(termination.fallen).tolist(),
            "out_of_bounds": np.asarray(termination.out_of_bounds).tolist(),
            "tagged": bool(np.asarray(termination.tagged)),
            "timed_out": bool(np.asarray(termination.timed_out)),
        },
    }
    config.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
