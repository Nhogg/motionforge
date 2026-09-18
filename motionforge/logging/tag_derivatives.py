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


class TagFootEvents(NamedTuple):
    """Foot planar speed plus slip and contact-transition events."""

    planar_speed: jax.Array
    slipping: jax.Array
    touchdown: jax.Array
    liftoff: jax.Array
    valid: jax.Array


class TagRecoveryEvents(NamedTuple):
    """Near-fall episode state, outcomes, and successful recovery duration."""

    active: jax.Array
    onset: jax.Array
    recovered: jax.Array
    failed: jax.Array
    duration: jax.Array


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


def derive_tag_foot_events(
    contact_active: jax.Array,
    position_world: jax.Array,
    control_timestep: float,
    slip_speed_threshold: float,
) -> TagFootEvents:
    """Derive foot speed, slip, touchdown, and liftoff from a trajectory."""
    if contact_active.ndim != 3 or contact_active.shape[1:] != (2, 2):
        raise ValueError(
            f"contact_active must have shape (T, 2, 2); got {contact_active.shape}"
        )
    if position_world.ndim != 4 or position_world.shape[1:] != (2, 2, 3):
        raise ValueError(
            f"position_world must have shape (T, 2, 2, 3); got {position_world.shape}"
        )
    if contact_active.shape[0] != position_world.shape[0]:
        raise ValueError("contact_active and position_world lengths must match")
    if contact_active.shape[0] == 0:
        raise ValueError("foot trajectories must contain at least one timestep")
    if control_timestep <= 0.0:
        raise ValueError("control_timestep must be positive")
    if slip_speed_threshold < 0.0:
        raise ValueError("slip_speed_threshold must be nonnegative")

    contact = jp.asarray(contact_active, dtype=bool)
    position = jp.asarray(position_world)
    planar_delta = position[1:, ..., :2] - position[:-1, ..., :2]
    speed = jp.linalg.norm(planar_delta, axis=-1) / control_timestep
    planar_speed = jp.concatenate([jp.zeros_like(speed[:1]), speed], axis=0)
    previous_contact = contact[:-1]
    current_contact = contact[1:]
    touchdown = jp.concatenate(
        [jp.zeros_like(contact[:1]), current_contact & ~previous_contact], axis=0
    )
    liftoff = jp.concatenate(
        [jp.zeros_like(contact[:1]), ~current_contact & previous_contact], axis=0
    )
    valid = jp.arange(contact.shape[0]) > 0
    slipping = contact & (planar_speed > slip_speed_threshold) & valid[:, None, None]
    return TagFootEvents(
        planar_speed=planar_speed,
        slipping=slipping,
        touchdown=touchdown,
        liftoff=liftoff,
        valid=valid,
    )


def derive_tag_recovery_events(
    near_fall: jax.Array,
    fallen: jax.Array,
    control_timestep: float,
) -> TagRecoveryEvents:
    """Segment near-fall warnings into successful or failed recovery episodes."""
    if near_fall.ndim != 2 or near_fall.shape[1:] != (2,):
        raise ValueError(f"near_fall must have shape (T, 2); got {near_fall.shape}")
    if fallen.shape != near_fall.shape:
        raise ValueError("fallen must have the same shape as near_fall")
    if near_fall.shape[0] == 0:
        raise ValueError("stability trajectories must contain at least one timestep")
    if control_timestep <= 0.0:
        raise ValueError("control_timestep must be positive")

    warning = jp.asarray(near_fall, dtype=bool)
    failure = jp.asarray(fallen, dtype=bool)
    timesteps = jp.arange(warning.shape[0], dtype=jp.int32)

    def scan_episode(carry, sample):
        active, start_step = carry
        step, current_warning, current_failure = sample
        onset = current_warning & ~active
        start_step = jp.where(onset, step, start_step)
        recovered = active & ~current_warning & ~current_failure
        failed = active & current_failure
        duration = jp.where(
            recovered,
            (step - start_step).astype(jp.float32) * control_timestep,
            0.0,
        )
        active = (active | current_warning) & ~recovered & ~failed
        return (active, start_step), (active, onset, recovered, failed, duration)

    (_, _), events = jax.lax.scan(
        scan_episode,
        (jp.zeros((2,), dtype=bool), jp.zeros((2,), dtype=jp.int32)),
        (timesteps, warning, failure),
    )
    return TagRecoveryEvents(*events)
