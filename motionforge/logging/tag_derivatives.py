"""Derived signals computed from recorded tag trajectories.

These transforms operate on complete time-major arrays after rollout capture.
They do not access simulator state or alter the online control loop.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


class TagLinearAcceleration(NamedTuple):
    """World-frame linear acceleration and per-timestep validity."""

    world: jax.Array
    valid: jax.Array


class TagYawAcceleration(NamedTuple):
    """Pelvis-frame yaw acceleration and per-timestep validity."""

    pelvis: jax.Array
    valid: jax.Array


class TagCommandDerivative(NamedTuple):
    """High-level velocity-command derivative and timestep validity."""

    velocity: jax.Array
    valid: jax.Array


class TagRollPitchExcursion(NamedTuple):
    """Signed root roll/pitch angles and their absolute excursions."""

    angle: jax.Array
    absolute: jax.Array


def derive_tag_linear_acceleration(
    world_linear_velocity: jax.Array,
    control_timestep: float,
) -> TagLinearAcceleration:
    """Differentiate time-major root velocity with a first-order difference."""
    if world_linear_velocity.ndim != 3 or world_linear_velocity.shape[1:] != (2, 3):
        raise ValueError(
            "world_linear_velocity must have shape (T, 2, 3); "
            f"got {world_linear_velocity.shape}"
        )
    if world_linear_velocity.shape[0] == 0:
        raise ValueError("world_linear_velocity must contain at least one timestep")
    if control_timestep <= 0.0:
        raise ValueError("control_timestep must be positive")

    velocity = jp.asarray(world_linear_velocity)
    differences = (velocity[1:] - velocity[:-1]) / control_timestep
    acceleration = jp.concatenate([jp.zeros_like(velocity[:1]), differences], axis=0)
    valid = jp.arange(velocity.shape[0]) > 0
    return TagLinearAcceleration(world=acceleration, valid=valid)


def derive_tag_yaw_acceleration(
    pelvis_angular_velocity: jax.Array,
    control_timestep: float,
) -> TagYawAcceleration:
    """Differentiate time-major pelvis yaw rate with a first-order difference."""
    if pelvis_angular_velocity.ndim != 3 or pelvis_angular_velocity.shape[1:] != (2, 3):
        raise ValueError(
            "pelvis_angular_velocity must have shape (T, 2, 3); "
            f"got {pelvis_angular_velocity.shape}"
        )
    if pelvis_angular_velocity.shape[0] == 0:
        raise ValueError("pelvis_angular_velocity must contain at least one timestep")
    if control_timestep <= 0.0:
        raise ValueError("control_timestep must be positive")

    yaw_rate = jp.asarray(pelvis_angular_velocity)[..., 2]
    differences = (yaw_rate[1:] - yaw_rate[:-1]) / control_timestep
    acceleration = jp.concatenate([jp.zeros_like(yaw_rate[:1]), differences], axis=0)
    valid = jp.arange(yaw_rate.shape[0]) > 0
    return TagYawAcceleration(pelvis=acceleration, valid=valid)


def derive_tag_command_velocity(
    command_velocity: jax.Array,
    control_timestep: float,
) -> TagCommandDerivative:
    """Differentiate time-major high-level commands."""
    if command_velocity.ndim != 3 or command_velocity.shape[1:] != (2, 3):
        raise ValueError(
            f"command_velocity must have shape (T, 2, 3); got {command_velocity.shape}"
        )
    if command_velocity.shape[0] == 0:
        raise ValueError("command_velocity must contain at least one timestep")
    if control_timestep <= 0.0:
        raise ValueError("control_timestep must be positive")

    command = jp.asarray(command_velocity)
    differences = (command[1:] - command[:-1]) / control_timestep
    derivative = jp.concatenate([jp.zeros_like(command[:1]), differences], axis=0)
    valid = jp.arange(command.shape[0]) > 0
    return TagCommandDerivative(velocity=derivative, valid=valid)


def derive_tag_roll_pitch_excursion(
    orientation_wxyz: jax.Array,
) -> TagRollPitchExcursion:
    """Convert time-major root quaternions to signed roll/pitch excursions."""
    if orientation_wxyz.ndim != 3 or orientation_wxyz.shape[1:] != (2, 4):
        raise ValueError(
            f"orientation_wxyz must have shape (T, 2, 4); got {orientation_wxyz.shape}"
        )
    if orientation_wxyz.shape[0] == 0:
        raise ValueError("orientation_wxyz must contain at least one timestep")

    quaternion = jp.asarray(orientation_wxyz)
    norm = jp.linalg.norm(quaternion, axis=-1, keepdims=True)
    normalized = quaternion / jp.where(norm > 0.0, norm, 1.0)
    w, x, y, z = jp.moveaxis(normalized, -1, 0)
    roll = jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    angle = jp.stack([roll, pitch], axis=-1)
    return TagRollPitchExcursion(angle=angle, absolute=jp.abs(angle))
