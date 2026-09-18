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
