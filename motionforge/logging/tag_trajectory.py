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
import mujoco
import numpy as np

from motionforge.envs.two_g1 import TwoG1Model

TAG_TRAJECTORY_SCHEMA_VERSION = 13


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


@dataclass(frozen=True)
class TagFootContactLayout:
    """Floor and left/right foot geom IDs for both agents."""

    floor_geom_id: int
    foot_geom_ids: tuple[tuple[int, int], tuple[int, int]]


class TagFootContact(NamedTuple):
    """Per-agent left/right floor contact and summed normal force."""

    active: jax.Array
    normal_force: jax.Array


class TagOpponentRelativeState(NamedTuple):
    """Opponent planar position and velocity in each agent's heading frame."""

    position: jax.Array
    velocity: jax.Array


@dataclass(frozen=True)
class TagFlatTerrainLayout:
    """Static world-frame description of the accepted flat floor."""

    height: float
    normal_world: tuple[float, float, float]


class TagTerrainState(NamedTuple):
    """Terrain height and world normal beneath both agents."""

    height: jax.Array
    normal_world: jax.Array


class TagRewardOutcome(NamedTuple):
    """Per-agent reward and canonical game termination outcome."""

    reward: jax.Array
    tagged: jax.Array
    fallen: jax.Array
    out_of_bounds: jax.Array
    timed_out: jax.Array
    done: jax.Array
    winner_index: jax.Array


@dataclass(frozen=True)
class TagTrackingLayout:
    """Sensor addresses needed for command tracking error."""

    pelvis_linear_velocity_slices: tuple[slice, slice]
    pelvis_angular_velocity_slices: tuple[slice, slice]


class TagTrackingError(NamedTuple):
    """Command-minus-measurement planar velocity and yaw-rate error."""

    velocity: jax.Array


class TagStabilityState(NamedTuple):
    """Canonical fall state and thresholded pre-fall warning state."""

    fallen: jax.Array
    near_fall: jax.Array
    root_height: jax.Array
    up_alignment: jax.Array


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


def build_tag_tracking_layout(model_bundle: TwoG1Model) -> TagTrackingLayout:
    """Resolve controller-frame velocity sensors once at setup time."""
    return TagTrackingLayout(
        pelvis_linear_velocity_slices=tuple(
            _sensor_slice(
                model_bundle.model,
                f"{agent.prefix}local_linvel_pelvis",
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


def build_tag_foot_contact_layout(model_bundle: TwoG1Model) -> TagFootContactLayout:
    """Resolve floor and foot collision geoms once at setup time."""
    return TagFootContactLayout(
        floor_geom_id=model_bundle.model.geom("floor").id,
        foot_geom_ids=tuple(
            (
                model_bundle.model.geom(f"{agent.prefix}left_foot").id,
                model_bundle.model.geom(f"{agent.prefix}right_foot").id,
            )
            for agent in model_bundle.agents
        ),
    )


def build_tag_flat_terrain_layout(model_bundle: TwoG1Model) -> TagFlatTerrainLayout:
    """Validate and describe the shared horizontal plane."""
    floor = model_bundle.model.geom("floor")
    if floor.type != mujoco.mjtGeom.mjGEOM_PLANE:
        raise RuntimeError("The P5 flat-terrain logger requires a plane floor geom")

    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, floor.quat)
    normal = rotation.reshape(3, 3)[:, 2]
    if not np.allclose(normal, np.asarray([0.0, 0.0, 1.0]), atol=1e-8):
        raise RuntimeError("The P5 flat-terrain logger requires a horizontal floor")

    return TagFlatTerrainLayout(
        height=float(floor.pos[2]),
        normal_world=tuple(float(value) for value in normal),
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


def extract_tag_foot_contact(data, layout: TagFootContactLayout) -> TagFootContact:
    """Extract Warp floor contacts and their normal constraint forces."""
    impl = data._impl
    contact_geoms = jp.asarray(impl.contact__geom)
    distances = jp.asarray(impl.contact__dist)
    world_ids = jp.asarray(impl.contact__worldid)
    active_counts = jp.asarray(impl.nacon)
    slots = jp.arange(distances.shape[0]) % impl.naconmax
    buffer_active = slots < active_counts[world_ids]

    normal_addresses = jp.asarray(impl.contact__efc_address)[:, 0]
    address_valid = normal_addresses >= 0
    safe_addresses = jp.clip(normal_addresses, 0, impl.efc__force.shape[0] - 1)
    normal_forces = jp.asarray(impl.efc__force)[safe_addresses]

    active_rows = []
    force_rows = []
    geom1 = contact_geoms[:, 0]
    geom2 = contact_geoms[:, 1]
    for agent_feet in layout.foot_geom_ids:
        active_feet = []
        foot_forces = []
        for foot_geom_id in agent_feet:
            foot_floor = ((geom1 == foot_geom_id) & (geom2 == layout.floor_geom_id)) | (
                (geom2 == foot_geom_id) & (geom1 == layout.floor_geom_id)
            )
            active = buffer_active & foot_floor & (distances <= 0.0)
            active_feet.append(jp.any(active))
            foot_forces.append(
                jp.sum(jp.where(active & address_valid, normal_forces, 0.0))
            )
        active_rows.append(jp.stack(active_feet))
        force_rows.append(jp.stack(foot_forces))

    return TagFootContact(
        active=jp.stack(active_rows),
        normal_force=jp.stack(force_rows),
    )


def tag_opponent_relative_state(
    position: jax.Array,
    velocity: jax.Array,
) -> TagOpponentRelativeState:
    """Validate canonical relative observations for trajectory logging."""
    if position.shape != (2, 2):
        raise ValueError(f"position must have shape (2, 2); got {position.shape}")
    if velocity.shape != (2, 2):
        raise ValueError(f"velocity must have shape (2, 2); got {velocity.shape}")
    return TagOpponentRelativeState(
        position=jp.asarray(position),
        velocity=jp.asarray(velocity),
    )


def extract_tag_flat_terrain(layout: TagFlatTerrainLayout) -> TagTerrainState:
    """Return the static flat-floor sample beneath both agents."""
    return TagTerrainState(
        height=jp.full((2,), layout.height),
        normal_world=jp.tile(jp.asarray(layout.normal_world), (2, 1)),
    )


def tag_reward_outcome(
    reward: jax.Array,
    tagged: jax.Array,
    fallen: jax.Array,
    out_of_bounds: jax.Array,
    timed_out: jax.Array,
    pursuer_index: int,
) -> TagRewardOutcome:
    """Validate rewards and classify canonical tag terminal outcomes."""
    if reward.shape != (2,):
        raise ValueError(f"reward must have shape (2,); got {reward.shape}")
    if fallen.shape != (2,):
        raise ValueError(f"fallen must have shape (2,); got {fallen.shape}")
    if out_of_bounds.shape != (2,):
        raise ValueError(
            f"out_of_bounds must have shape (2,); got {out_of_bounds.shape}"
        )
    if pursuer_index not in (0, 1):
        raise ValueError("pursuer_index must be 0 or 1")

    evader_index = 1 - pursuer_index
    failed = jp.asarray(fallen) | jp.asarray(out_of_bounds)
    one_failed = failed[0] ^ failed[1]
    failure_winner = jp.where(failed[0], 1, 0)
    winner_index = jp.asarray(-1, dtype=jp.int32)
    winner_index = jp.where(one_failed, failure_winner, winner_index)
    winner_index = jp.where(timed_out, evader_index, winner_index)
    winner_index = jp.where(tagged, pursuer_index, winner_index)
    done = tagged | jp.any(failed) | timed_out

    return TagRewardOutcome(
        reward=jp.asarray(reward),
        tagged=jp.asarray(tagged),
        fallen=jp.asarray(fallen),
        out_of_bounds=jp.asarray(out_of_bounds),
        timed_out=jp.asarray(timed_out),
        done=done,
        winner_index=winner_index,
    )


def extract_tag_tracking_error(
    data,
    layout: TagTrackingLayout,
    command_velocity: jax.Array,
) -> TagTrackingError:
    """Compute command-minus-measurement error in controller coordinates."""
    if command_velocity.shape != (2, 3):
        raise ValueError(
            f"command_velocity must have shape (2, 3); got {command_velocity.shape}"
        )

    measured = []
    for linear_slice, angular_slice in zip(
        layout.pelvis_linear_velocity_slices,
        layout.pelvis_angular_velocity_slices,
        strict=True,
    ):
        local_linear = data.sensordata[linear_slice]
        local_angular = data.sensordata[angular_slice]
        measured.append(
            jp.asarray([local_linear[0], local_linear[1], local_angular[2]])
        )

    return TagTrackingError(
        velocity=jp.asarray(command_velocity) - jp.stack(measured),
    )


def tag_stability_state(
    fallen: jax.Array,
    root_height: jax.Array,
    up_alignment: jax.Array,
    near_fall_minimum_root_height: float,
    near_fall_minimum_up_alignment: float,
) -> TagStabilityState:
    """Classify near falls while preserving the environment's fall flag."""
    if fallen.shape != (2,):
        raise ValueError(f"fallen must have shape (2,); got {fallen.shape}")
    if root_height.shape != (2,):
        raise ValueError(f"root_height must have shape (2,); got {root_height.shape}")
    if up_alignment.shape != (2,):
        raise ValueError(f"up_alignment must have shape (2,); got {up_alignment.shape}")
    if near_fall_minimum_root_height <= 0.0:
        raise ValueError("near_fall_minimum_root_height must be positive")
    if not 0.0 <= near_fall_minimum_up_alignment <= 1.0:
        raise ValueError("near_fall_minimum_up_alignment must be in [0, 1]")

    fallen = jp.asarray(fallen)
    root_height = jp.asarray(root_height)
    up_alignment = jp.asarray(up_alignment)
    near_fall = (~fallen) & (
        (root_height < near_fall_minimum_root_height)
        | (up_alignment < near_fall_minimum_up_alignment)
    )
    return TagStabilityState(
        fallen=fallen,
        near_fall=near_fall,
        root_height=root_height,
        up_alignment=up_alignment,
    )
