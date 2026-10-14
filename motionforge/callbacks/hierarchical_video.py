"""Scheduled current-policy TAG videos for hierarchical training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import jax
import mediapy
import mujoco
import numpy as np
from mujoco import mjx

from motionforge.logging.wandb import WandbMetricsLogger
from motionforge.rollout import HierarchicalRolloutRunner, HierarchicalRolloutSegment


@dataclass
class HierarchicalTagVideoCallback:
    runner: HierarchicalRolloutRunner
    logger: WandbMetricsLogger
    output_dir: Path
    interval_steps: int
    duration: float = 10.0
    fps: int = 20
    width: int = 960
    height: int = 720
    seed: int = 0
    render_step_zero: bool = True
    _last_rendered_step: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.interval_steps <= 0:
            raise ValueError("video interval must be positive")

    def __call__(self, environment_steps: int, parameters: Any) -> None:
        if self._last_rendered_step is None:
            should_render = self.render_step_zero or environment_steps > 0
        else:
            should_render = (
                environment_steps - self._last_rendered_step >= self.interval_steps
            )
        if not should_render:
            return
        for role in ("pursuer", "evader"):
            path = self.output_dir / "videos" / f"{role}_{environment_steps:012d}.mp4"
            metadata = self._render(parameters, role, path)
            metadata["environment_steps"] = environment_steps
            path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.logger.log_video(
                environment_steps=environment_steps,
                path=path,
                caption=f"current self-play policy as {role}, step={environment_steps}",
                key=f"evaluation/{role}_policy_video",
            )
        self._last_rendered_step = environment_steps

    def _render(self, parameters: Any, role: str, output: Path) -> dict[str, Any]:
        learner_is_pursuer = role == "pursuer"
        learner_index = (
            self.runner.environment.config.pursuer_index
            if learner_is_pursuer
            else 1 - self.runner.environment.config.pursuer_index
        )
        segment = HierarchicalRolloutSegment(
            learner_parameters=parameters,
            opponent_parameters=parameters,
            opponent_id="current/self-play",
            opponent_category="recent",
            learner_index=learner_index,
            learner_is_pursuer=learner_is_pursuer,
        )
        state = self.runner.reset(jax.random.PRNGKey(self.seed))
        step = jax.jit(self.runner.step)
        model = self.runner.environment.model_bundle.model
        colors = (np.asarray([0.20, 0.75, 0.30]), np.asarray([0.90, 0.35, 0.15]))
        geom_groups = (
            self.runner.environment.contact_layout.agent0_geom_ids,
            self.runner.environment.contact_layout.agent1_geom_ids,
        )
        for agent_index, geom_ids in enumerate(geom_groups):
            color = colors[0] if agent_index == learner_index else colors[1]
            for geom_id in geom_ids:
                model.geom_rgba[geom_id, :3] = color
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
        renderer = mujoco.Renderer(model, width=self.width, height=self.height)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 5.0
        camera.azimuth = 135.0
        camera.elevation = -20.0
        frames: list[np.ndarray] = []
        next_frame_time = 0.0
        control_timestep = self.runner.environment.config.control_timestep
        rollout_steps = round(self.duration / control_timestep)
        terminated_at = None
        try:
            for index in range(rollout_steps):
                state = step(segment, state)
                state.environment.physics.data.qpos.block_until_ready()
                simulation_time = (index + 1) * control_timestep
                positions = np.asarray(state.environment.diagnostics.planar_position)
                if simulation_time >= next_frame_time:
                    camera.lookat[:2] = positions.mean(axis=0)
                    camera.lookat[2] = 0.7
                    host_data = mjx.get_data(model, state.environment.physics.data)
                    renderer.update_scene(host_data, camera=camera)
                    frames.append(renderer.render().copy())
                    next_frame_time += 1.0 / self.fps
                if bool(np.asarray(state.environment.done)):
                    terminated_at = simulation_time
                    break
        finally:
            renderer.close()
        if not frames:
            raise RuntimeError("no hierarchical video frames were captured")
        output.parent.mkdir(parents=True, exist_ok=True)
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        mediapy.write_video(output, frames, fps=self.fps)
        return {
            "duration": len(frames) / self.fps,
            "frames": len(frames),
            "learner_role": role,
            "output": str(output.resolve()),
            "seed": self.seed,
            "terminated_at": terminated_at,
        }
