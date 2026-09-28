"""Policy-agnostic raw state samples for two-agent TAG comparisons.

The extractor performs no inference, simulation, serialization, or host
transfer. Dataset generators provide the commands and low-level controls that
produced the current shared-physics state.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from motionforge.envs.tag_environment import TagEnvironmentState
from motionforge.envs.two_g1 import TwoG1Model

TAG_STATE_DATASET_SCHEMA_VERSION = 1


class TagStateSample(NamedTuple):
    """Comparable raw physical and task state for both G1 agents."""

    root_position_world: jax.Array
    root_orientation_wxyz: jax.Array
    root_generalized_linear_velocity: jax.Array
    root_generalized_angular_velocity: jax.Array
    joint_position: jax.Array
    joint_velocity: jax.Array
    command_velocity: jax.Array
    normalized_action: jax.Array
    joint_position_target: jax.Array
    relative_position: jax.Array
    relative_velocity: jax.Array
    arena_center_position: jax.Array
    agent_distance: jax.Array
    root_height: jax.Array
    up_alignment: jax.Array
    fallen: jax.Array
    near_fall: jax.Array
    out_of_bounds: jax.Array
    tag_contact: jax.Array
    tag_contact_count: jax.Array
    tagged: jax.Array
    timed_out: jax.Array


def extract_tag_state_sample(
    model_bundle: TwoG1Model,
    state: TagEnvironmentState,
    command_velocity: jax.Array,
    normalized_action: jax.Array,
    joint_position_target: jax.Array,
    *,
    near_fall_minimum_root_height: float = 0.60,
    near_fall_minimum_up_alignment: float = 0.80,
) -> TagStateSample:
    """Extract one two-agent record using stable model-layout slices."""
    if command_velocity.shape != (2, 3):
        raise ValueError("command_velocity must have shape (2, 3)")
    if normalized_action.shape != (2, 29):
        raise ValueError("normalized_action must have shape (2, 29)")
    if joint_position_target.shape != (2, 29):
        raise ValueError("joint_position_target must have shape (2, 29)")
    if near_fall_minimum_root_height <= 0.0:
        raise ValueError("near_fall_minimum_root_height must be positive")
    if not 0.0 <= near_fall_minimum_up_alignment <= 1.0:
        raise ValueError("near_fall_minimum_up_alignment must be in [0, 1]")

    qpos = tuple(state.data.qpos[agent.qpos_slice] for agent in model_bundle.agents)
    qvel = tuple(state.data.qvel[agent.qvel_slice] for agent in model_bundle.agents)
    root_position = jp.stack([value[:3] for value in qpos])
    root_orientation = jp.stack([value[3:7] for value in qpos])
    joint_position = jp.stack([value[7:] for value in qpos])
    root_linear_velocity = jp.stack([value[:3] for value in qvel])
    root_angular_velocity = jp.stack([value[3:6] for value in qvel])
    joint_velocity = jp.stack([value[6:] for value in qvel])

    fallen = state.termination.fallen | state.diagnostics.fall_detected
    near_fall = ~fallen & (
        (state.diagnostics.root_height < near_fall_minimum_root_height)
        | (state.diagnostics.up_alignment < near_fall_minimum_up_alignment)
    )
    distance = jp.linalg.norm(state.observation.relative_position[0])

    return TagStateSample(
        root_position_world=root_position,
        root_orientation_wxyz=root_orientation,
        root_generalized_linear_velocity=root_linear_velocity,
        root_generalized_angular_velocity=root_angular_velocity,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        command_velocity=jp.asarray(command_velocity),
        normalized_action=jp.asarray(normalized_action),
        joint_position_target=jp.asarray(joint_position_target),
        relative_position=state.observation.relative_position,
        relative_velocity=state.observation.relative_velocity,
        arena_center_position=state.observation.arena_center_position,
        agent_distance=distance,
        root_height=state.diagnostics.root_height,
        up_alignment=state.diagnostics.up_alignment,
        fallen=fallen,
        near_fall=near_fall,
        out_of_bounds=state.termination.out_of_bounds
        | state.diagnostics.out_of_bounds,
        tag_contact=state.diagnostics.tag_contact,
        tag_contact_count=state.diagnostics.tag_contact_count,
        tagged=state.termination.tagged,
        timed_out=state.termination.timed_out,
    )
