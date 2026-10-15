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

from motionforge.envs.tag_endings import tag_ending_metadata
from motionforge.evaluation.hierarchical_roles import (
    annotate_hierarchical_frame,
    audit_hierarchical_roles,
    audit_hierarchical_role_trace,
    color_hierarchical_agents,
)
from motionforge.evaluation.hierarchical_trajectory import (
    plot_path,
    trajectory_diagnostics,
)
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
            metadata["run_name"] = self.output_dir.name
            plot = path.with_suffix(".png")
            plot_path(metadata, plot)
            path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.logger.log_video(
                environment_steps=environment_steps,
                path=path,
                caption=(
                    f"green=pursuer, orange=evader; learner={role}; "
                    f"current self-play, step={environment_steps}"
                ),
                key=f"evaluation/{role}_policy_video",
            )
            self.logger.log_image(
                environment_steps=environment_steps,
                path=plot,
                caption=(
                    f"Same self-play rollout as {role} video; "
                    "green=pursuer, orange=evader"
                ),
                key=f"evaluation/{role}_trajectory",
            )
            self.logger.log(
                environment_steps=environment_steps,
                metrics={
                    f"evaluation/{role}_rollout/tagged": float(
                        metadata["termination"]["tagged"]
                    ),
                    f"evaluation/{role}_rollout/pursuer_out_of_bounds": float(
                        metadata["termination"]["out_of_bounds"][
                            metadata["pursuer_index"]
                        ]
                    ),
                    f"evaluation/{role}_rollout/final_distance": metadata[
                        "final_distance"
                    ],
                    f"evaluation/{role}_rollout/pursuer_toward_evader_mps": metadata[
                        "mean_pursuer_toward_evader_mps"
                    ],
                    f"evaluation/{role}_rollout/pursuer_moving_away_fraction": metadata[
                        "pursuer_moving_away_fraction"
                    ],
                },
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
        audit = audit_hierarchical_roles(
            pursuer_index=self.runner.environment.config.pursuer_index,
            segment=segment,
            roles=state.roles,
        )
        step = jax.jit(self.runner.step)
        model = self.runner.environment.model_bundle.model
        colored_geom_counts = color_hierarchical_agents(
            self.runner.environment.model_bundle, pursuer_index=audit["pursuer_index"]
        )
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
        positions_history = [np.asarray(state.environment.diagnostics.planar_position)]
        role_history = [np.asarray(state.roles)]
        try:
            for index in range(rollout_steps):
                state = step(segment, state)
                state.environment.physics.data.qpos.block_until_ready()
                simulation_time = (index + 1) * control_timestep
                positions = np.asarray(state.environment.diagnostics.planar_position)
                positions_history.append(positions)
                role_history.append(np.asarray(state.roles))
                if simulation_time >= next_frame_time:
                    camera.lookat[:2] = positions.mean(axis=0)
                    camera.lookat[2] = 0.7
                    host_data = mjx.get_data(model, state.environment.physics.data)
                    renderer.update_scene(host_data, camera=camera)
                    frames.append(
                        annotate_hierarchical_frame(
                            renderer.render(),
                            audit,
                            stochastic_actions=self.runner.config.stochastic_actions,
                            stochastic_locomotion_actions=(
                                self.runner.config.sample_locomotion_actions
                            ),
                            matchup="current self-play",
                        )
                    )
                    next_frame_time += 1.0 / self.fps
                if bool(np.asarray(state.environment.done)):
                    terminated_at = simulation_time
                    break
        finally:
            renderer.close()
        if not frames:
            raise RuntimeError("no hierarchical video frames were captured")
        role_trace_audit = audit_hierarchical_role_trace(
            np.stack(role_history),
            pursuer_index=audit["pursuer_index"],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        mediapy.write_video(output, frames, fps=self.fps)
        termination = state.environment.termination
        return {
            **audit,
            **role_trace_audit,
            **trajectory_diagnostics(positions_history, control_timestep, audit),
            "arena_half_extent": self.runner.environment.config.arena_half_extent,
            "termination": {
                **tag_ending_metadata(termination, collection_cutoff=True),
                "tagged": bool(np.asarray(termination.tagged)),
                "fallen": np.asarray(termination.fallen).tolist(),
                "out_of_bounds": np.asarray(termination.out_of_bounds).tolist(),
                "timed_out": bool(np.asarray(termination.timed_out)),
            },
            "colored_geom_counts": colored_geom_counts,
            "opponent_id": segment.opponent_id,
            "stochastic_actions": self.runner.config.stochastic_actions,
            "stochastic_locomotion_actions": (
                self.runner.config.sample_locomotion_actions
            ),
            "strategy_action_mode": (
                "stochastic"
                if self.runner.config.stochastic_actions
                else "deterministic"
            ),
            "motor_action_mode": (
                "stochastic"
                if self.runner.config.sample_locomotion_actions
                else "deterministic"
            ),
            "initial_heading_noise_radians": getattr(
                self.runner.environment.config, "heading_noise_radians", 0.0
            ),
            "duration": len(frames) / self.fps,
            "frames": len(frames),
            "learner_role": role,
            "output": str(output.resolve()),
            "seed": self.seed,
            "terminated_at": terminated_at,
        }
