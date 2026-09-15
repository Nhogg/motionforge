"""tag_environment.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Integrated MJX-Warp runtime for the two-G1 tag environment.

    The environment owns shared physics, deterministic reset state, and
    episode bookeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jp
from flax import struct
from mujoco import mjx

from motionforge.envs.tag_bounds import (
    BoundsDetectionConfig,
    build_bounds_detection_layout,
    detect_out_of_bounds,
)
from motionforge.envs.tag_contact import (
    build_tag_contact_layout,
    detect_tag_contact,
)
from motionforge.envs.tag_fall import (
    FallDetectionConfig,
    build_fall_detection_layout,
    detect_falls,
)
from motionforge.envs.tag_observations import (
    build_tag_observation_layout,
    relative_planar_observation,
)
from motionforge.envs.tag_reset import (
    TagResetConfig,
    build_tag_reset_layout,
    sample_tag_reset,
)
from motionforge.envs.tag_roles import (
    TagRoleAssignment,
    assign_tag_roles,
)
from motionforge.envs.tag_timeout import (
    EpisodeTimeoutConfig,
    observe_episode_timeout,
)
from motionforge.envs.two_g1 import (
    TwoG1Model,
    build_two_g1_model,
    make_two_g1_data,
)


@dataclass(frozen=True)
class TagEnvironmentConfig:
    simulation_timestep: float = 0.002
    control_timestep: float = 0.02
    episode_duration: float = 20.0
    separation: float = 2.0
    arena_half_extent: float = 4.0
    minimum_root_height: float = 0.45
    minimum_up_alignment: float = 0.50
    fall_persistence_steps: int = 5
    pursuer_index: int = 0
    naconmax: int = 64
    njmax: int = 256

    def __post_init__(self) -> None:
        if self.simulation_timestep <= 0.0:
            raise ValueError("simulation_timestep must be positive")
        if self.control_timestep <= 0.0:
            raise ValueError("control_timestep must be positive")

        substeps = self.control_timestep / self.simulation_timestep
        if not math.isclose(substeps, round(substeps), abs_tol=1e-9):
            raise ValueError(
                "control_timestep must be an integer multiple of simulation_timestep"
            )
        if self.fall_persistence_steps <= 0:
            raise ValueError("fall_persistence_steps must be positive")
        if self.naconmax <= 0:
            raise ValueError("naconmax must be positive")
        if self.njmax <= 0:
            raise ValueError("njmax must be positive")

        TagResetConfig(separation=self.separation)
        BoundsDetectionConfig(arena_half_extent=self.arena_half_extent)
        FallDetectionConfig(
            minimum_root_height=self.minimum_root_height,
            minimum_up_alignment=self.minimum_up_alignment,
        )
        EpisodeTimeoutConfig(
            duration=self.episode_duration,
            control_timestep=self.control_timestep,
        )

    @property
    def physics_substeps(self) -> int:
        return round(self.control_timestep / self.simulation_timestep)


@struct.dataclass
class TagEnvironmentObservation:
    """Per-agent observations of the other agent in local heading frames."""

    relative_position: jax.Array
    relative_velocity: jax.Array


@struct.dataclass
class TagEnvironmentDiagnostics:
    """Current rule measurements retained for logging and reward design."""

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
class TagEnvironmentTermination:
    """Sticky terminal causes accumulated over an episode."""

    tagged: jax.Array
    fallen: jax.Array
    out_of_bounds: jax.Array
    timed_out: jax.Array


@struct.dataclass
class TagEnvironmentState:
    data: mjx.Data
    observation: TagEnvironmentObservation
    diagnostics: TagEnvironmentDiagnostics
    termination: TagEnvironmentTermination
    rng: jax.Array
    step_count: jax.Array
    fall_counts: jax.Array
    done: jax.Array


class TwoG1TagEnvironment:
    """Shared two-agent MJX-Warp physics and episode state."""

    def __init__(
        self,
        config: TagEnvironmentConfig | None = None,
    ) -> None:
        if config is None:
            config = TagEnvironmentConfig()
        self.config = config

        self.model_bundle: TwoG1Model = build_two_g1_model(
            timestep=config.simulation_timestep,
            enable_inter_agent_collision=True,
        )
        native_data = make_two_g1_data(
            self.model_bundle,
            separation=config.separation,
        )

        self.model = mjx.put_model(
            self.model_bundle.model,
            impl="warp",
        )
        self.template_data = mjx.put_data(
            self.model_bundle.model,
            native_data,
            impl="warp",
            naconmax=config.naconmax,
            njmax=config.njmax,
        )

        self.roles: TagRoleAssignment = assign_tag_roles(
            self.model_bundle.agents,
            pursuer_index=config.pursuer_index,
        )

        self.reset_layout = build_tag_reset_layout(self.model_bundle)
        self.observation_layout = build_tag_observation_layout(self.model_bundle)
        self.contact_layout = build_tag_contact_layout(self.model_bundle)
        self.fall_layout = build_fall_detection_layout(self.model_bundle)
        self.bounds_layout = build_bounds_detection_layout(self.model_bundle)

        self.reset_config = TagResetConfig(
            separation=config.separation,
        )
        self.fall_config = FallDetectionConfig(
            minimum_root_height=config.minimum_root_height,
            minimum_up_alignment=config.minimum_up_alignment,
        )
        self.bounds_config = BoundsDetectionConfig(
            arena_half_extent=config.arena_half_extent,
        )
        self.timeout_config = EpisodeTimeoutConfig(
            duration=config.episode_duration,
            control_timestep=config.control_timestep,
        )

        actuator_ids = jp.asarray(
            [agent.actuator_ids for agent in self.model_bundle.agents],
            dtype=jp.int32,
        )
        self.actuator_ids = actuator_ids
        self.default_joint_targets = self.template_data.ctrl[actuator_ids]

    @property
    def action_shape(self) -> tuple[int, int]:
        return (2, 29)

    def _get_observation(self, data: mjx.Data) -> TagEnvironmentObservation:
        observations = tuple(
            relative_planar_observation(
                data,
                self.observation_layout,
                observer_index,
            )
            for observer_index in range(2)
        )
        return TagEnvironmentObservation(
            relative_position=jp.stack(
                [observation.position for observation in observations]
            ),
            relative_velocity=jp.stack(
                [observation.velocity for observation in observations]
            ),
        )

    def _get_diagnostics(
        self,
        data: mjx.Data,
        step_count: jax.Array,
    ) -> TagEnvironmentDiagnostics:
        contact = detect_tag_contact(data, self.contact_layout)
        falls = detect_falls(data, self.fall_layout, self.fall_config)
        bounds = detect_out_of_bounds(data, self.bounds_layout, self.bounds_config)
        timeout = observe_episode_timeout(step_count, self.timeout_config)

        return TagEnvironmentDiagnostics(
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

    def reset(self, key: jax.Array) -> TagEnvironmentState:
        """Create a deterministic episode state from an explicit key."""
        reset_key, next_key = jax.random.split(key)
        sampled = sample_tag_reset(
            self.template_data.qpos,
            self.template_data.qvel,
            reset_key,
            self.reset_layout,
            self.reset_config,
        )

        data = self.template_data.replace(
            qpos=sampled.qpos,
            qvel=sampled.qvel,
        )
        data = mjx.forward(self.model, data)
        step_count = jp.zeros((), dtype=jp.int32)
        diagnostics = self._get_diagnostics(data, step_count)

        return TagEnvironmentState(
            data=data,
            observation=self._get_observation(data),
            diagnostics=diagnostics,
            termination=TagEnvironmentTermination(
                tagged=jp.zeros((), dtype=bool),
                fallen=jp.zeros((2,), dtype=bool),
                out_of_bounds=jp.zeros((2,), dtype=bool),
                timed_out=jp.zeros((), dtype=bool),
            ),
            rng=next_key,
            step_count=step_count,
            fall_counts=jp.zeros((2,), dtype=jp.int32),
            done=jp.zeros((), dtype=bool),
        )

    def step(
        self,
        state: TagEnvironmentState,
        joint_position_targets: jax.Array,
    ) -> TagEnvironmentState:
        """Advance one control update using per-agent joint targets."""
        if joint_position_targets.shape != self.action_shape:
            raise ValueError(
                "joint_position_targets must have shape "
                f"{self.action_shape}; got {joint_position_targets.shape}"
            )

        controls = state.data.ctrl.at[self.actuator_ids].set(joint_position_targets)
        data = state.data.replace(ctrl=controls)

        def physics_step(current_data, _):
            next_data = mjx.step(self.model, current_data)
            return next_data, None

        data, _ = jax.lax.scan(
            physics_step,
            data,
            xs=None,
            length=self.config.physics_substeps,
        )

        step_count = state.step_count + 1
        diagnostics = self._get_diagnostics(data, step_count)
        fall_counts = jp.where(
            diagnostics.fall_detected,
            state.fall_counts + 1,
            0,
        )
        persistent_falls = fall_counts >= self.config.fall_persistence_steps
        termination = TagEnvironmentTermination(
            tagged=state.termination.tagged | diagnostics.tag_contact,
            fallen=state.termination.fallen | persistent_falls,
            out_of_bounds=(
                state.termination.out_of_bounds | diagnostics.out_of_bounds
            ),
            timed_out=state.termination.timed_out | diagnostics.timed_out,
        )
        done = (
            termination.tagged
            | jp.any(termination.fallen)
            | jp.any(termination.out_of_bounds)
            | termination.timed_out
        )

        return state.replace(
            data=data,
            observation=self._get_observation(data),
            diagnostics=diagnostics,
            termination=termination,
            step_count=step_count,
            fall_counts=fall_counts,
            done=done,
        )
