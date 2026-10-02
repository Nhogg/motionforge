"""Boundary-aware filtering for high-level tag velocity commands.

The filter accepts a heading-frame ``[forward, lateral, yaw]`` command and
projects only its world-frame planar components that point farther outside a
square arena. It is stateless, JAX-compatible, and independent of policy and
locomotion-controller implementations.
"""

from __future__ import annotations

import jax
import jax.numpy as jp


def heading_planar_to_world(
    vector: jax.Array,
    root_quaternion: jax.Array,
) -> jax.Array:
    """Rotate a root-yaw-frame planar vector into the world frame."""
    if vector.shape != (2,):
        raise ValueError(f"vector must have shape (2,), got {vector.shape}")
    if root_quaternion.shape != (4,):
        raise ValueError(
            "root_quaternion must have shape (4,), "
            f"got {root_quaternion.shape}"
        )
    w, x, y, z = root_quaternion
    yaw = jp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    cosine = jp.cos(yaw)
    sine = jp.sin(yaw)
    return jp.asarray(
        [
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ]
    )


def world_planar_to_heading(
    vector: jax.Array,
    root_quaternion: jax.Array,
) -> jax.Array:
    """Rotate a world-frame planar vector into the root yaw frame."""
    if vector.shape != (2,):
        raise ValueError(f"vector must have shape (2,), got {vector.shape}")
    if root_quaternion.shape != (4,):
        raise ValueError(
            "root_quaternion must have shape (4,), "
            f"got {root_quaternion.shape}"
        )
    w, x, y, z = root_quaternion
    yaw = jp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    cosine = jp.cos(yaw)
    sine = jp.sin(yaw)
    return jp.asarray(
        [
            cosine * vector[0] + sine * vector[1],
            -sine * vector[0] + cosine * vector[1],
        ]
    )


def project_boundary_safe_command(
    command: jax.Array,
    planar_position: jax.Array,
    root_quaternion: jax.Array,
    *,
    arena_half_extent: float,
    boundary_margin: float,
) -> jax.Array:
    """Smoothly remove outward command velocity inside a square edge margin.

    At the inner edge of the margin the command is unchanged. Its outward
    component is attenuated linearly until it reaches zero at the arena edge.
    Inward and edge-tangential components, along with yaw rate, are preserved.
    """
    if command.shape != (3,):
        raise ValueError(f"command must have shape (3,), got {command.shape}")
    if planar_position.shape != (2,):
        raise ValueError(
            f"planar_position must have shape (2,), got {planar_position.shape}"
        )
    if arena_half_extent <= 0.0:
        raise ValueError("arena_half_extent must be positive")
    if boundary_margin <= 0.0 or boundary_margin > arena_half_extent:
        raise ValueError(
            "boundary_margin must be positive and not exceed arena_half_extent"
        )

    world_velocity = heading_planar_to_world(command[:2], root_quaternion)
    safe_half_extent = arena_half_extent - boundary_margin
    axis_intrusion = jp.clip(
        (jp.abs(planar_position) - safe_half_extent) / boundary_margin,
        0.0,
        1.0,
    )
    outward_direction = jp.sign(planar_position)
    outward_speed = jp.maximum(outward_direction * world_velocity, 0.0)
    safe_world_velocity = (
        world_velocity
        - outward_direction * axis_intrusion * outward_speed
    )
    safe_heading_velocity = world_planar_to_heading(
        safe_world_velocity,
        root_quaternion,
    )
    return jp.concatenate([safe_heading_velocity, command[2:3]])
