"""tag_fall.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Fall classification for agents in the two-G1 tag environment.

    Inputs are generalized positions from the shared two-agent model. The detector
    reports root height, torso-up alignment, finite-state validity, and a fall flag
    for each agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp

from motionforge.envs.two_g1 import TwoG1Model


@dataclass(frozen=True)
class FallDetectionConfig:
    minimum_root_height: float = 0.45
    minimum_up_alignment: float = 0.50

    def __post_init__(self) -> None:
        if self.minimum_root_height <= 0.0:
            raise ValueError("minimum_root_height must be positive")
        if not -1.0 <= self.minimum_up_alignment <= 1.0:
            raise ValueError("minimum_up_alignment must be between -1 and 1")


@dataclass(frozen=True)
class AgentFallLayout:
    root_qpos_start: int


@dataclass(frozen=True)
class FallDetectionLayout:
    agents: tuple[AgentFallLayout, AgentFallLayout]


class FallObservation(NamedTuple):
    fallen: jax.Array
    root_height: jax.Array
    up_alignment: jax.Array
    state_finite: jax.Array


def build_fall_detection_layout(
    model_bundle: TwoG1Model,
) -> FallDetectionLayout:
    """Resolve each agent's floating-base position address."""
    return FallDetectionLayout(
        agents=tuple(
            AgentFallLayout(root_qpos_start=agent.qpos_slice.start)
            for agent in model_bundle.agents
        )
    )


def _quaternion_up_alignment(quaternion: jax.Array) -> jax.Array:
    """Return local-Z alignment with world-Z for a wxyz quaternion."""
    w, x, y, z = quaternion
    norm_squared = jp.maximum(
        jp.dot(quaternion, quaternion),
        jp.finfo(quaternion.dtype).eps,
    )
    return (w * w - x * x - y * y + z * z) / norm_squared


def detect_falls(
    data,
    layout: FallDetectionLayout,
    config: FallDetectionConfig = FallDetectionConfig(),
) -> FallObservation:
    """Classify each agent independently from root height and orientation."""
    qpos = jp.asarray(data.qpos)

    root_heights = jp.stack(
        [qpos[agent.root_qpos_start + 2] for agent in layout.agents]
    )
    root_quaternions = jp.stack(
        [
            qpos[agent.root_qpos_start + 3 : agent.root_qpos_start + 7]
            for agent in layout.agents
        ]
    )

    up_alignments = jax.vmap(_quaternion_up_alignment)(root_quaternions)
    state_finite = jp.isfinite(root_heights) & jp.all(
        jp.isfinite(root_quaternions),
        axis=-1,
    )

    fallen = (
        ~state_finite
        | (root_heights < config.minimum_root_height)
        | (up_alignments < config.minimum_up_alignment)
    )

    return FallObservation(
        fallen=fallen,
        root_height=root_heights,
        up_alignment=up_alignments,
        state_finite=state_finite,
    )
