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

TAG_TRAJECTORY_SCHEMA_VERSION = 2


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
