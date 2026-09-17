"""Extract versioned trajectory records from the two-agent tag state.

The extractors in this module are independent of simulation stepping, policy
inference, and file I/O. Host-side experiments decide when and where records
are serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp

from motionforge.envs.two_g1 import TwoG1Model

TAG_TRAJECTORY_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class TagRootPoseLayout:
    """Generalized-position addresses for both floating bases."""

    root_qpos_starts: tuple[int, int]


class TagRootPose(NamedTuple):
    """World-frame root position and wxyz orientation for both agents."""

    position: jax.Array
    orientation_wxyz: jax.Array


@dataclass(frozen=True)
class TagRootVelocityLayout:
    """Sensor addresses for both agents' root velocities."""

    world_linear_velocity_slices: tuple[slice, slice]
    pelvis_angular_velocity_slices: tuple[slice, slice]


class TagRootVelocity(NamedTuple):
    """World linear and pelvis-frame angular velocity for both agents."""

    world_linear: jax.Array
    pelvis_angular: jax.Array


@dataclass(frozen=True)
class TagJointStateLayout:
    """Generalized-coordinate slices for both agents' actuated joints."""

    joint_qpos_slices: tuple[slice, slice]
    joint_qvel_slices: tuple[slice, slice]


class TagJointState(NamedTuple):
    """Raw joint positions and velocities for both agents."""

    position: jax.Array
    velocity: jax.Array


class TagControllerCommand(NamedTuple):
    """High-level velocity commands and applied low-level joint targets."""

    velocity: jax.Array
    joint_position_target: jax.Array


def build_tag_root_pose_layout(model_bundle: TwoG1Model) -> TagRootPoseLayout:
    """Resolve root generalized-position addresses once at setup time."""
    return TagRootPoseLayout(
        root_qpos_starts=tuple(agent.qpos_slice.start for agent in model_bundle.agents)
    )


def _sensor_slice(model, name: str, expected_dimension: int) -> slice:
    sensor_id = model.sensor(name).id
    start = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])
    if dimension != expected_dimension:
        raise RuntimeError(
            f"Sensor {name!r} has dimension {dimension}; expected {expected_dimension}"
        )
    return slice(start, start + dimension)


def build_tag_root_velocity_layout(
    model_bundle: TwoG1Model,
) -> TagRootVelocityLayout:
    """Resolve velocity sensor addresses once at setup time."""
    return TagRootVelocityLayout(
        world_linear_velocity_slices=tuple(
            _sensor_slice(
                model_bundle.model,
                f"{agent.prefix}global_linvel_pelvis",
                3,
            )
            for agent in model_bundle.agents
        ),
        pelvis_angular_velocity_slices=tuple(
            _sensor_slice(
                model_bundle.model,
                f"{agent.prefix}gyro_pelvis",
                3,
            )
            for agent in model_bundle.agents
        ),
    )


def build_tag_joint_state_layout(model_bundle: TwoG1Model) -> TagJointStateLayout:
    """Resolve the 29 actuated joint coordinates for each agent."""
    return TagJointStateLayout(
        joint_qpos_slices=tuple(
            slice(agent.qpos_slice.start + 7, agent.qpos_slice.stop)
            for agent in model_bundle.agents
        ),
        joint_qvel_slices=tuple(
            slice(agent.qvel_slice.start + 6, agent.qvel_slice.stop)
            for agent in model_bundle.agents
        ),
    )


def extract_tag_root_pose(data, layout: TagRootPoseLayout) -> TagRootPose:
    """Extract both root poses without transferring state off device."""
    positions = []
    orientations = []
    for root_start in layout.root_qpos_starts:
        positions.append(data.qpos[root_start : root_start + 3])
        orientations.append(data.qpos[root_start + 3 : root_start + 7])

    return TagRootPose(
        position=jp.stack(positions),
        orientation_wxyz=jp.stack(orientations),
    )


def extract_tag_root_velocity(
    data,
    layout: TagRootVelocityLayout,
) -> TagRootVelocity:
    """Extract explicitly framed root velocities without a host transfer."""
    return TagRootVelocity(
        world_linear=jp.stack(
            [
                data.sensordata[sensor_slice]
                for sensor_slice in layout.world_linear_velocity_slices
            ]
        ),
        pelvis_angular=jp.stack(
            [
                data.sensordata[sensor_slice]
                for sensor_slice in layout.pelvis_angular_velocity_slices
            ]
        ),
    )


def extract_tag_joint_state(data, layout: TagJointStateLayout) -> TagJointState:
    """Extract both agents' actuated joint state without a host transfer."""
    return TagJointState(
        position=jp.stack(
            [data.qpos[joint_slice] for joint_slice in layout.joint_qpos_slices]
        ),
        velocity=jp.stack(
            [data.qvel[joint_slice] for joint_slice in layout.joint_qvel_slices]
        ),
    )


def tag_controller_command(
    velocity: jax.Array,
    joint_position_target: jax.Array,
) -> TagControllerCommand:
    """Validate externally supplied controller inputs for trajectory logging."""
    if velocity.shape != (2, 3):
        raise ValueError(f"velocity must have shape (2, 3); got {velocity.shape}")
    if joint_position_target.shape != (2, 29):
        raise ValueError(
            "joint_position_target must have shape (2, 29); "
            f"got {joint_position_target.shape}"
        )
    return TagControllerCommand(
        velocity=jp.asarray(velocity),
        joint_position_target=jp.asarray(joint_position_target),
    )
