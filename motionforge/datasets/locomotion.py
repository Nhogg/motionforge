"""Common single-agent locomotion samples for baseline and TAG datasets.

The extractor reads simulator and controller state without performing policy
inference, stepping physics, or writing files. Dataset generators own episode
management and serialization.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

LOCOMOTION_DATASET_SCHEMA_VERSION = 1


def _quaternion_up_alignment(quaternion: jax.Array) -> jax.Array:
    w, x, y, z = quaternion
    norm_squared = jp.maximum(
        jp.dot(quaternion, quaternion),
        jp.finfo(quaternion.dtype).eps,
    )
    return (w * w - x * x - y * y + z * z) / norm_squared


class G1LocomotionSample(NamedTuple):
    """Raw state and applied control for one G1 control timestep."""

    root_position_world: jax.Array
    root_orientation_wxyz: jax.Array
    root_linear_velocity_world: jax.Array
    root_linear_velocity_local: jax.Array
    pelvis_angular_velocity: jax.Array
    joint_position: jax.Array
    joint_velocity: jax.Array
    command_velocity: jax.Array
    command_tracking_error: jax.Array
    normalized_action: jax.Array
    joint_position_target: jax.Array
    foot_contact: jax.Array
    foot_position_world: jax.Array
    root_height: jax.Array
    up_alignment: jax.Array
    fallen: jax.Array
    near_fall: jax.Array


def extract_g1_locomotion_sample(
    environment,
    state,
    command_velocity: jax.Array,
    normalized_action: jax.Array,
    joint_position_target: jax.Array,
    *,
    fall_minimum_root_height: float,
    fall_minimum_up_alignment: float,
    near_fall_minimum_root_height: float,
    near_fall_minimum_up_alignment: float,
) -> G1LocomotionSample:
    """Extract one comparable single-agent record without a host transfer."""
    if command_velocity.shape != (3,):
        raise ValueError(
            f"command_velocity must have shape (3,); got {command_velocity.shape}"
        )
    if normalized_action.shape != (29,):
        raise ValueError(
            f"normalized_action must have shape (29,); got {normalized_action.shape}"
        )
    if joint_position_target.shape != (29,):
        raise ValueError(
            "joint_position_target must have shape (29,); "
            f"got {joint_position_target.shape}"
        )

    world_velocity = environment.get_global_linvel(state.data, "pelvis")
    local_velocity = environment.get_local_linvel(state.data, "pelvis")
    angular_velocity = environment.get_gyro(state.data, "pelvis")
    root_height = state.data.qpos[2]
    up_alignment = _quaternion_up_alignment(state.data.qpos[3:7])
    fallen = (root_height < fall_minimum_root_height) | (
        up_alignment < fall_minimum_up_alignment
    )
    near_fall = ~fallen & (
        (root_height < near_fall_minimum_root_height)
        | (up_alignment < near_fall_minimum_up_alignment)
    )
    measured_command = jp.asarray(
        [local_velocity[0], local_velocity[1], angular_velocity[2]]
    )

    return G1LocomotionSample(
        root_position_world=state.data.qpos[:3],
        root_orientation_wxyz=state.data.qpos[3:7],
        root_linear_velocity_world=world_velocity,
        root_linear_velocity_local=local_velocity,
        pelvis_angular_velocity=angular_velocity,
        joint_position=state.data.qpos[7:],
        joint_velocity=state.data.qvel[6:],
        command_velocity=jp.asarray(command_velocity),
        command_tracking_error=jp.asarray(command_velocity) - measured_command,
        normalized_action=jp.asarray(normalized_action),
        joint_position_target=jp.asarray(joint_position_target),
        foot_contact=jp.asarray(state.info["last_contact"], dtype=bool),
        foot_position_world=state.data.site_xpos[environment._feet_site_id],
        root_height=root_height,
        up_alignment=up_alignment,
        fallen=fallen,
        near_fall=near_fall,
    )
