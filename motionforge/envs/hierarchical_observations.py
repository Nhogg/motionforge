"""Observation adapters between multi-agent Warp state and the hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp

from motionforge.controllers.g1_tag import G1TagPolicyObservationLayout
from motionforge.envs.multi_agent_task import MultiAgentTaskState
from motionforge.models import LocomotionObservation, StrategyObservation


@dataclass(frozen=True)
class HierarchicalObservationConfig:
    """Scaling inherited from the accepted split TAG policies."""

    arena_half_extent: float = 4.0
    relative_velocity_scale: float = 2.0
    command_scale: tuple[float, float, float] = (1.0, 0.5, 1.0)

    def __post_init__(self) -> None:
        if self.arena_half_extent <= 0.0:
            raise ValueError("arena_half_extent must be positive")
        if self.relative_velocity_scale <= 0.0:
            raise ValueError("relative_velocity_scale must be positive")
        if len(self.command_scale) != 3:
            raise ValueError("command_scale must contain three values")
        if any(scale <= 0.0 for scale in self.command_scale):
            raise ValueError("command scales must be positive")


def role_encodings(pursuer_index: jax.Array) -> jax.Array:
    """Return per-agent ``[pursuer, evader]`` one-hot role encodings."""
    pursuer_index = jp.asarray(pursuer_index, dtype=jp.int32)
    agent_indices = jp.arange(2, dtype=jp.int32)
    pursuer = agent_indices == pursuer_index
    return jp.stack((pursuer, ~pursuer), axis=-1).astype(jp.float32)


def strategy_observations(
    state: MultiAgentTaskState,
    previous_commands: jax.Array,
    roles: jax.Array,
    config: HierarchicalObservationConfig,
) -> StrategyObservation:
    """Build normalized task-level observations for both agents."""
    if previous_commands.shape != (2, 3):
        raise ValueError("previous_commands must have shape (2, 3)")
    if roles.shape != (2, 2):
        raise ValueError("roles must have shape (2, 2)")
    command_scale = jp.asarray(config.command_scale)
    return StrategyObservation(
        opponent_position_local=(
            state.observation.relative_position / (2.0 * config.arena_half_extent)
        ),
        opponent_velocity_local=(
            state.observation.relative_velocity / config.relative_velocity_scale
        ),
        arena_center_local=(
            state.observation.arena_center_position / config.arena_half_extent
        ),
        previous_command=previous_commands / command_scale,
        role=roles,
    )


def locomotion_observations(
    state: MultiAgentTaskState,
    layout: G1TagPolicyObservationLayout,
    previous_actions: jax.Array,
    phases: jax.Array,
    default_poses: jax.Array,
) -> LocomotionObservation:
    """Build P2-compatible motor-level observation fields for both agents."""
    if previous_actions.shape != (2, 29):
        raise ValueError("previous_actions must have shape (2, 29)")
    if phases.shape != (2, 2):
        raise ValueError("phases must have shape (2, 2)")
    if default_poses.shape != (2, 29):
        raise ValueError("default_poses must have shape (2, 29)")

    data = state.physics.data
    root_linear_velocity = []
    root_angular_velocity = []
    projected_gravity = []
    joint_position = []
    joint_velocity = []
    for agent_index, agent in enumerate(layout.agents):
        root_linear_velocity.append(
            data.sensordata[agent.local_velocity_sensor_slice]
        )
        root_angular_velocity.append(data.sensordata[agent.gyro_sensor_slice])
        projected_gravity.append(
            data.site_xmat[agent.pelvis_imu_site_id].T
            @ jp.asarray([0.0, 0.0, -1.0])
        )
        joint_position.append(
            data.qpos[agent.qpos_slice.start + 7 : agent.qpos_slice.stop]
            - default_poses[agent_index]
        )
        joint_velocity.append(
            data.qvel[agent.qvel_slice.start + 6 : agent.qvel_slice.stop]
        )

    return LocomotionObservation(
        root_linear_velocity=jp.stack(root_linear_velocity),
        root_angular_velocity=jp.stack(root_angular_velocity),
        projected_gravity=jp.stack(projected_gravity),
        joint_position=jp.stack(joint_position),
        joint_velocity=jp.stack(joint_velocity),
        previous_action=previous_actions,
        phase_cosine=jp.cos(phases),
        phase_sine=jp.sin(phases),
    )
