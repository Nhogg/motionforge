"""Role-conditioned tag task layered over league-enabled Warp physics."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp
from flax import struct
from mujoco import mjx

from motionforge.envs.league import LeagueEnv
from motionforge.envs.tag_bounds import (
    BoundsDetectionConfig,
    build_bounds_detection_layout,
    detect_out_of_bounds,
)
from motionforge.envs.tag_contact import build_tag_contact_layout, detect_tag_contact
from motionforge.envs.tag_fall import (
    FallDetectionConfig,
    build_fall_detection_layout,
    detect_falls,
)
from motionforge.envs.tag_observations import (
    build_tag_observation_layout,
    relative_planar_observation,
    world_planar_to_heading,
)
from motionforge.envs.tag_roles import TagRoleAssignment, assign_tag_roles
from motionforge.envs.tag_timeout import (
    EpisodeTimeoutConfig,
    observe_episode_timeout,
)
from motionforge.envs.warp import WarpEnvConfig, WarpEnvState
from motionforge.league import League


@dataclass(frozen=True)
class MultiAgentTaskConfig(WarpEnvConfig):
    episode_duration: float = 20.0
    arena_half_extent: float = 4.0
    minimum_root_height: float = 0.45
    minimum_up_alignment: float = 0.50
    fall_persistence_steps: int = 5
    pursuer_index: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fall_persistence_steps <= 0:
            raise ValueError("fall_persistence_steps must be positive")
        if self.pursuer_index not in (0, 1):
            raise ValueError("pursuer_index must be 0 or 1")
        BoundsDetectionConfig(arena_half_extent=self.arena_half_extent)
        FallDetectionConfig(
            minimum_root_height=self.minimum_root_height,
            minimum_up_alignment=self.minimum_up_alignment,
        )
        EpisodeTimeoutConfig(
            duration=self.episode_duration,
            control_timestep=self.control_timestep,
        )


@struct.dataclass
class MultiAgentTaskObservation:
    relative_position: jax.Array
    relative_velocity: jax.Array
    arena_center_position: jax.Array


@struct.dataclass
class MultiAgentTaskDiagnostics:
    tag_contact: jax.Array
    tag_contact_count: jax.Array
    tag_minimum_distance: jax.Array
    fall_detected: jax.Array
    root_height: jax.Array
    up_alignment: jax.Array
    root_state_finite: jax.Array
    out_of_bounds: jax.Array
    planar_position: jax.Array
    boundary_excess: jax.Array
    planar_state_finite: jax.Array
    timed_out: jax.Array
    elapsed_seconds: jax.Array


@struct.dataclass
class MultiAgentTaskTermination:
    tagged: jax.Array
    fallen: jax.Array
    out_of_bounds: jax.Array
    timed_out: jax.Array


@struct.dataclass
class MultiAgentTaskState:
    physics: WarpEnvState
    observation: MultiAgentTaskObservation
    diagnostics: MultiAgentTaskDiagnostics
    termination: MultiAgentTaskTermination
    fall_counts: jax.Array
    done: jax.Array


class MultiAgentTaskEnv(LeagueEnv):
    """Two-role tag task using the required league-enabled environment path."""

    def __init__(
        self,
        *,
        league: League,
        config: MultiAgentTaskConfig | None = None,
    ) -> None:
        task_config = MultiAgentTaskConfig() if config is None else config
        super().__init__(league=league, config=task_config)
        self.config = task_config
        self.roles: TagRoleAssignment = assign_tag_roles(
            self.model_bundle.agents,
            pursuer_index=task_config.pursuer_index,
        )
        self.observation_layout = build_tag_observation_layout(self.model_bundle)
        self.contact_layout = build_tag_contact_layout(self.model_bundle)
        self.fall_layout = build_fall_detection_layout(self.model_bundle)
        self.bounds_layout = build_bounds_detection_layout(self.model_bundle)
        self.fall_config = FallDetectionConfig(
            minimum_root_height=task_config.minimum_root_height,
            minimum_up_alignment=task_config.minimum_up_alignment,
        )
        self.bounds_config = BoundsDetectionConfig(
            arena_half_extent=task_config.arena_half_extent,
        )
        self.timeout_config = EpisodeTimeoutConfig(
            duration=task_config.episode_duration,
            control_timestep=task_config.control_timestep,
        )

    def _get_observation(self, data: mjx.Data) -> MultiAgentTaskObservation:
        relative = tuple(
            relative_planar_observation(data, self.observation_layout, index)
            for index in range(2)
        )
        return MultiAgentTaskObservation(
            relative_position=jp.stack([value.position for value in relative]),
            relative_velocity=jp.stack([value.velocity for value in relative]),
            arena_center_position=jp.stack(
                [
                    world_planar_to_heading(
                        -data.xpos[agent.root_body_id, :2],
                        data.qpos[
                            agent.root_qpos_start + 3 : agent.root_qpos_start + 7
                        ],
                    )
                    for agent in self.observation_layout.agents
                ]
            ),
        )

    def _get_diagnostics(
        self,
        data: mjx.Data,
        step_count: jax.Array,
    ) -> MultiAgentTaskDiagnostics:
        contact = detect_tag_contact(data, self.contact_layout)
        falls = detect_falls(data, self.fall_layout, self.fall_config)
        bounds = detect_out_of_bounds(data, self.bounds_layout, self.bounds_config)
        timeout = observe_episode_timeout(step_count, self.timeout_config)
        return MultiAgentTaskDiagnostics(
            tag_contact=contact.occurred,
            tag_contact_count=contact.count,
            tag_minimum_distance=contact.minimum_distance,
            fall_detected=falls.fallen,
            root_height=falls.root_height,
            up_alignment=falls.up_alignment,
            root_state_finite=falls.state_finite,
            out_of_bounds=bounds.out_of_bounds,
            planar_position=bounds.planar_position,
            boundary_excess=bounds.boundary_excess,
            planar_state_finite=bounds.state_finite,
            timed_out=timeout.timed_out,
            elapsed_seconds=timeout.elapsed_seconds,
        )

    def reset(self, key: jax.Array) -> MultiAgentTaskState:
        physics = super().reset(key)
        diagnostics = self._get_diagnostics(physics.data, physics.step_count)
        return MultiAgentTaskState(
            physics=physics,
            observation=self._get_observation(physics.data),
            diagnostics=diagnostics,
            termination=MultiAgentTaskTermination(
                tagged=jp.zeros((), dtype=bool),
                fallen=jp.zeros((2,), dtype=bool),
                out_of_bounds=jp.zeros((2,), dtype=bool),
                timed_out=jp.zeros((), dtype=bool),
            ),
            fall_counts=jp.zeros((2,), dtype=jp.int32),
            done=jp.zeros((), dtype=bool),
        )

    def step(
        self,
        state: MultiAgentTaskState,
        joint_position_targets: jax.Array,
    ) -> MultiAgentTaskState:
        physics = super().step(state.physics, joint_position_targets)
        diagnostics = self._get_diagnostics(physics.data, physics.step_count)
        fall_counts = jp.where(
            diagnostics.fall_detected,
            state.fall_counts + 1,
            0,
        )
        persistent_falls = fall_counts >= self.config.fall_persistence_steps
        termination = MultiAgentTaskTermination(
            tagged=state.termination.tagged | diagnostics.tag_contact,
            fallen=state.termination.fallen | persistent_falls,
            out_of_bounds=state.termination.out_of_bounds
            | diagnostics.out_of_bounds,
            timed_out=state.termination.timed_out | diagnostics.timed_out,
        )
        done = (
            termination.tagged
            | jp.any(termination.fallen)
            | jp.any(termination.out_of_bounds)
            | termination.timed_out
        )
        return state.replace(
            physics=physics,
            observation=self._get_observation(physics.data),
            diagnostics=diagnostics,
            termination=termination,
            fall_counts=fall_counts,
            done=done,
        )
