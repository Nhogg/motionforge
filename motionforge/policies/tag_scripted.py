"""Scripted high-level policies for the two-agent tag task.

Policies in this module consume environment observations and emit G1 velocity
commands. They do not own physics or the learned low-level locomotion policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp


@dataclass(frozen=True)
class ScriptedPursuerConfig:
    """Gains and command limits for direct pursuit in the local heading frame."""

    linear_gain: float = 0.8
    yaw_gain: float = 1.5
    tag_radius: float = 0.35
    maximum_forward_speed: float = 0.8
    maximum_lateral_speed: float = 0.4
    maximum_yaw_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.linear_gain < 0.0:
            raise ValueError("linear_gain must be nonnegative")
        if self.yaw_gain < 0.0:
            raise ValueError("yaw_gain must be nonnegative")
        if self.tag_radius < 0.0:
            raise ValueError("tag_radius must be nonnegative")
        if self.maximum_forward_speed <= 0.0:
            raise ValueError("maximum_forward_speed must be positive")
        if self.maximum_lateral_speed <= 0.0:
            raise ValueError("maximum_lateral_speed must be positive")
        if self.maximum_yaw_rate <= 0.0:
            raise ValueError("maximum_yaw_rate must be positive")


_DEFAULT_PURSUER_CONFIG = ScriptedPursuerConfig()


@dataclass(frozen=True)
class ScriptedEvaderConfig:
    """Gains and command limits for moving away in the local heading frame."""

    linear_gain: float = 0.8
    yaw_gain: float = 1.5
    maximum_forward_speed: float = 0.5
    maximum_lateral_speed: float = 0.25
    maximum_yaw_rate: float = 0.6
    arena_radius: float = 4.0
    boundary_margin: float = 1.5
    boundary_gain: float = 0.5

    def __post_init__(self) -> None:
        if self.linear_gain < 0.0:
            raise ValueError("linear_gain must be nonnegative")
        if self.yaw_gain < 0.0:
            raise ValueError("yaw_gain must be nonnegative")
        if self.maximum_forward_speed <= 0.0:
            raise ValueError("maximum_forward_speed must be positive")
        if self.maximum_lateral_speed <= 0.0:
            raise ValueError("maximum_lateral_speed must be positive")
        if self.maximum_yaw_rate <= 0.0:
            raise ValueError("maximum_yaw_rate must be positive")
        if self.arena_radius <= 0.0:
            raise ValueError("arena_radius must be positive")
        if not 0.0 < self.boundary_margin <= self.arena_radius:
            raise ValueError("boundary_margin must be in (0, arena_radius]")
        if self.boundary_gain < 0.0:
            raise ValueError("boundary_gain must be nonnegative")


_DEFAULT_EVADER_CONFIG = ScriptedEvaderConfig()


def scripted_pursuer_command(
    relative_position: jax.Array,
    config: ScriptedPursuerConfig = _DEFAULT_PURSUER_CONFIG,
) -> jax.Array:
    """Return ``[forward velocity, lateral velocity, yaw rate]`` toward a target."""
    if relative_position.shape != (2,):
        raise ValueError(
            f"relative_position must have shape (2,), got {relative_position.shape}"
        )

    relative_position = jp.asarray(relative_position, dtype=jp.float32)
    distance = jp.linalg.norm(relative_position)
    bearing = jp.arctan2(relative_position[1], relative_position[0])

    translation = jp.asarray(
        [
            jp.clip(
                config.linear_gain * relative_position[0],
                -config.maximum_forward_speed,
                config.maximum_forward_speed,
            ),
            jp.clip(
                config.linear_gain * relative_position[1],
                -config.maximum_lateral_speed,
                config.maximum_lateral_speed,
            ),
        ]
    )
    translation = jp.where(distance <= config.tag_radius, 0.0, translation)
    yaw_rate = jp.clip(
        config.yaw_gain * bearing,
        -config.maximum_yaw_rate,
        config.maximum_yaw_rate,
    )

    return jp.concatenate([translation, yaw_rate[None]])


def scripted_evader_command(
    relative_position: jax.Array,
    arena_center_position: jax.Array,
    config: ScriptedEvaderConfig = _DEFAULT_EVADER_CONFIG,
) -> jax.Array:
    """Return ``[forward velocity, lateral velocity, yaw rate]`` away from a threat."""
    if relative_position.shape != (2,):
        raise ValueError(
            f"relative_position must have shape (2,), got {relative_position.shape}"
        )
    if arena_center_position.shape != (2,):
        raise ValueError(
            "arena_center_position must have shape (2,), "
            f"got {arena_center_position.shape}"
        )

    relative_position = jp.asarray(relative_position, dtype=jp.float32)
    arena_center_position = jp.asarray(arena_center_position, dtype=jp.float32)
    escape_direction = -relative_position
    center_distance = jp.linalg.norm(arena_center_position)
    safe_radius = config.arena_radius - config.boundary_margin
    inward_direction = arena_center_position / jp.maximum(
        center_distance,
        jp.finfo(arena_center_position.dtype).eps,
    )
    escape_velocity = config.linear_gain * escape_direction
    inward_velocity = config.boundary_gain * inward_direction
    desired_velocity = jp.where(
        center_distance >= safe_radius,
        inward_velocity,
        escape_velocity,
    )
    desired_bearing = jp.arctan2(desired_velocity[1], desired_velocity[0])

    translation = jp.asarray(
        [
            jp.clip(
                desired_velocity[0],
                -config.maximum_forward_speed,
                config.maximum_forward_speed,
            ),
            jp.clip(
                desired_velocity[1],
                -config.maximum_lateral_speed,
                config.maximum_lateral_speed,
            ),
        ]
    )
    yaw_rate = jp.clip(
        config.yaw_gain * desired_bearing,
        -config.maximum_yaw_rate,
        config.maximum_yaw_rate,
    )

    return jp.concatenate([translation, yaw_rate[None]])
