"""Deterministic reset sampling for the two-G1 tag environment.

The reset preserves the accepted G1 joint pose, zeros generalized velocities,
and uses an explicit JAX key to rotate a symmetric, face-to-face spawn around
the arena origin. It returns arrays for direct insertion into MJX-Warp data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp

from motionforge.envs.two_g1 import TwoG1Model


@dataclass(frozen=True)
class TagResetConfig:
    separation: float = 2.0
    minimum_separation: float | None = None
    maximum_separation: float | None = None
    slope_degrees: float = 0.0
    slope_curriculum_degrees: float = 0.0
    heading_noise_radians: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.heading_noise_radians)
            or not 0 <= self.heading_noise_radians <= math.pi
        ):
            raise ValueError("heading_noise_radians must be finite and within [0, pi]")
        if self.separation <= 0.0:
            raise ValueError("separation must be positive")
        minimum = (
            self.separation
            if self.minimum_separation is None
            else self.minimum_separation
        )
        maximum = (
            self.separation
            if self.maximum_separation is None
            else self.maximum_separation
        )
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError("separation range must be positive and ordered")
        if not math.isfinite(self.slope_degrees) or abs(self.slope_degrees) > 30.0:
            raise ValueError("slope_degrees must be finite and within [-30, 30]")
        if (
            not math.isfinite(self.slope_curriculum_degrees)
            or self.slope_curriculum_degrees < 0.0
            or self.slope_curriculum_degrees > 30.0
        ):
            raise ValueError("slope_curriculum_degrees must lie within [0, 30]")
        if self.slope_degrees != 0.0 and self.slope_curriculum_degrees != 0.0:
            raise ValueError("fixed slope and slope curriculum are mutually exclusive")


_DEFAULT_TAG_RESET_CONFIG = TagResetConfig()


@dataclass(frozen=True)
class AgentResetLayout:
    root_qpos_start: int
    joint_qpos_slice: slice


@dataclass(frozen=True)
class TagResetLayout:
    agents: tuple[AgentResetLayout, AgentResetLayout]


class TagResetState(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array
    spawn_angle: jax.Array
    planar_positions: jax.Array
    headings: jax.Array
    slope_degrees: jax.Array
    floor_quaternion: jax.Array


def build_tag_reset_layout(model_bundle: TwoG1Model) -> TagResetLayout:
    """Resolve floating-base and actuated-joint position addresses."""
    return TagResetLayout(
        agents=tuple(
            AgentResetLayout(
                root_qpos_start=agent.qpos_slice.start,
                joint_qpos_slice=slice(
                    agent.qpos_slice.start + 7,
                    agent.qpos_slice.stop,
                ),
            )
            for agent in model_bundle.agents
        )
    )


def sample_tag_reset(
    template_qpos: jax.Array,
    template_qvel: jax.Array,
    key: jax.Array,
    layout: TagResetLayout,
    config: TagResetConfig = _DEFAULT_TAG_RESET_CONFIG,
) -> TagResetState:
    """Sample a reproducible symmetric spawn from an explicit random key."""
    qpos = jp.asarray(template_qpos)
    qvel = jp.zeros_like(jp.asarray(template_qvel))

    angle_key, separation_key = jax.random.split(key)
    spawn_angle = jax.random.uniform(
        angle_key,
        shape=(),
        minval=-jp.pi,
        maxval=jp.pi,
    )
    spawn_direction = jp.asarray([jp.cos(spawn_angle), jp.sin(spawn_angle)])
    minimum_separation = (
        config.separation
        if config.minimum_separation is None
        else config.minimum_separation
    )
    maximum_separation = (
        config.separation
        if config.maximum_separation is None
        else config.maximum_separation
    )
    separation = (
        jp.asarray(minimum_separation)
        if minimum_separation == maximum_separation
        else jax.random.uniform(
            separation_key,
            shape=(),
            minval=minimum_separation,
            maxval=maximum_separation,
        )
    )
    planar_positions = jp.stack(
        [
            -0.5 * separation * spawn_direction,
            0.5 * separation * spawn_direction,
        ]
    )
    headings = jp.asarray(
        [
            spawn_angle,
            jp.arctan2(
                jp.sin(spawn_angle + jp.pi),
                jp.cos(spawn_angle + jp.pi),
            ),
        ]
    )

    if config.heading_noise_radians > 0.0:
        # Independent robot headings, without changing the original spawn or
        # separation streams. Default zero preserves the old reset exactly.
        headings = headings + jax.random.uniform(
            jax.random.fold_in(key, 2),
            shape=(2,),
            minval=-config.heading_noise_radians,
            maxval=config.heading_noise_radians,
        )
        headings = jp.arctan2(jp.sin(headings), jp.cos(headings))

    if config.slope_curriculum_degrees > 0.0:
        terrain_index = jax.random.randint(
            jax.random.fold_in(key, 1),
            shape=(),
            minval=0,
            maxval=3,
        )
        slope_degrees = jp.asarray(
            [
                -config.slope_curriculum_degrees,
                0.0,
                config.slope_curriculum_degrees,
            ]
        )[terrain_index]
    else:
        slope_degrees = jp.asarray(config.slope_degrees)
    half_slope = 0.5 * jp.deg2rad(slope_degrees)
    floor_quaternion = jp.asarray([jp.cos(half_slope), 0.0, jp.sin(half_slope), 0.0])

    for agent_index, agent in enumerate(layout.agents):
        root_start = agent.root_qpos_start
        half_heading = 0.5 * headings[agent_index]
        root_quaternion = jp.asarray(
            [
                jp.cos(half_heading),
                0.0,
                0.0,
                jp.sin(half_heading),
            ]
        )
        qpos = qpos.at[root_start : root_start + 2].set(planar_positions[agent_index])
        terrain_height = (
            -jp.tan(jp.deg2rad(slope_degrees)) * (planar_positions[agent_index, 0])
        )
        qpos = qpos.at[root_start + 2].add(terrain_height)
        qpos = qpos.at[root_start + 3 : root_start + 7].set(root_quaternion)

    return TagResetState(
        qpos=qpos,
        qvel=qvel,
        spawn_angle=spawn_angle,
        planar_positions=planar_positions,
        headings=headings,
        slope_degrees=slope_degrees,
        floor_quaternion=floor_quaternion,
    )
