"""Deterministic reset sampling for the two-G1 tag environment.

The reset preserves the accepted G1 joint pose, zeros generalized velocities,
and uses an explicit JAX key to rotate a symmetric, face-to-face spawn around
the arena origin. It returns arrays for direct insertion into MJX-Warp data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp

from motionforge.envs.two_g1 import TwoG1Model


@dataclass(frozen=True)
class TagResetConfig:
    separation: float = 2.0

    def __post_init__(self) -> None:
        if self.separation <= 0.0:
            raise ValueError("separation must be positive")


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

    spawn_angle = jax.random.uniform(
        key,
        shape=(),
        minval=-jp.pi,
        maxval=jp.pi,
    )
    spawn_direction = jp.asarray(
        [jp.cos(spawn_angle), jp.sin(spawn_angle)]
    )
    planar_positions = jp.stack(
        [
            -0.5 * config.separation * spawn_direction,
            0.5 * config.separation * spawn_direction,
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
        qpos = qpos.at[root_start : root_start + 2].set(
            planar_positions[agent_index]
        )
        qpos = qpos.at[root_start + 3 : root_start + 7].set(root_quaternion)

    return TagResetState(
        qpos=qpos,
        qvel=qvel,
        spawn_angle=spawn_angle,
        planar_positions=planar_positions,
        headings=headings,
    )
