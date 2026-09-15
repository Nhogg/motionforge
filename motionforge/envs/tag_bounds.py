"""Out-of-bounds detection for the two-G1 tag environment.

The playable area is an axis-aligned square centered on the world origin.
Detection uses each agent's floating-base planar position and operates on array
state compatible with the MJX-Warp runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp

from motionforge.envs.two_g1 import TwoG1Model


@dataclass(frozen=True)
class BoundsDetectionConfig:
    arena_half_extent: float = 4.0

    def __post_init__(self) -> None:
        if self.arena_half_extent <= 0.0:
            raise ValueError("arena_half_extent must be positive")


@dataclass(frozen=True)
class AgentBoundsLayout:
    root_qpos_start: int


@dataclass(frozen=True)
class BoundsDetectionLayout:
    agents: tuple[AgentBoundsLayout, AgentBoundsLayout]


class BoundsObservation(NamedTuple):
    out_of_bounds: jax.Array
    planar_position: jax.Array
    boundary_excess: jax.Array
    state_finite: jax.Array


def build_bounds_detection_layout(
    model_bundle: TwoG1Model,
) -> BoundsDetectionLayout:
    """Resolve each agent's floating-base position address."""
    return BoundsDetectionLayout(
        agents=tuple(
            AgentBoundsLayout(root_qpos_start=agent.qpos_slice.start)
            for agent in model_bundle.agents
        )
    )


def detect_out_of_bounds(
    data,
    layout: BoundsDetectionLayout,
    config: BoundsDetectionConfig = BoundsDetectionConfig(),
) -> BoundsObservation:
    """Classify agents outside the square playable area."""
    qpos = jp.asarray(data.qpos)
    planar_positions = jp.stack(
        [
            qpos[agent.root_qpos_start : agent.root_qpos_start + 2]
            for agent in layout.agents
        ]
    )

    state_finite = jp.all(jp.isfinite(planar_positions), axis=-1)
    boundary_excess = jp.max(
        jp.abs(planar_positions) - config.arena_half_extent,
        axis=-1,
    )
    out_of_bounds = ~state_finite | (boundary_excess > 0.0)

    return BoundsObservation(
        out_of_bounds=out_of_bounds,
        planar_position=planar_positions,
        boundary_excess=boundary_excess,
        state_finite=state_finite,
    )
