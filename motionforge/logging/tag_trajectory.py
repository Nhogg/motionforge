"""Extract versioned trajectory records from the two-agent tag state.

The extractors in this module are independent of simulation stepping, policy
inference, and file I/O. Host-side experiments decide when and where records
are serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp

from motionforge.envs.two_g1 import TwoG1Model

TAG_TRAJECTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TagRootPoseLayout:
    """Generalized-position addresses for both floating bases."""

    root_qpos_starts: tuple[int, int]


class TagRootPose(NamedTuple):
    """World-frame root position and wxyz orientation for both agents."""

    position: jax.Array
    orientation_wxyz: jax.Array


def build_tag_root_pose_layout(model_bundle: TwoG1Model) -> TagRootPoseLayout:
    """Resolve root generalized-position addresses once at setup time."""
    return TagRootPoseLayout(
        root_qpos_starts=tuple(agent.qpos_slice.start for agent in model_bundle.agents)
    )


def extract_tag_root_pose(data, layout: TagRootPoseLayout) -> TagRootPose:
    """Extract both root poses without transferring state off device."""
    positions = []
    orientations = []
    for root_start in layout.root_qpos_starts:
        positions.append(data.qpos[root_start : root_start + 3])
        orientations.append(data.qpos[root_start + 3 : root_start + 7])

    return TagRootPose(
        position=jp.stack(positions),
        orientation_wxyz=jp.stack(orientations),
    )
