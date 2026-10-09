"""Learner-only transitions collected from hierarchical two-agent rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp
from flax import struct

from motionforge.envs import (
    TagEvaderConfig,
    TagPursuerConfig,
    evader_reward,
    locomotion_observations,
    pursuer_reward,
    strategy_observations,
)
from motionforge.rollout.hierarchical import (
    HierarchicalRolloutRunner,
    HierarchicalRolloutSegment,
    HierarchicalRolloutState,
)


@dataclass(frozen=True)
class LocomotionRewardConfig:
    linear_tracking_weight: float = 1.0
    yaw_tracking_weight: float = 0.5
    upright_weight: float = 0.25
    action_rate_weight: float = 0.01
    joint_deviation_weight: float = 0.005
    fall_penalty: float = 5.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")


@struct.dataclass
class LocomotionRewardTerms:
    linear_tracking: jax.Array
    yaw_tracking: jax.Array
    upright: jax.Array
    action_rate: jax.Array
    joint_deviation: jax.Array
    fall: jax.Array
    total: jax.Array


@struct.dataclass
class HierarchicalTransition:
    """One 50 Hz learner transition with strategy and motor supervision."""

    strategy_observation: jax.Array
    strategy_command: jax.Array
    strategy_value: jax.Array
    strategy_log_probability: jax.Array
    strategy_entropy: jax.Array
    strategy_reward: jax.Array
    strategy_update: jax.Array
    locomotion_observation: jax.Array
    locomotion_action: jax.Array
    locomotion_value: jax.Array
    locomotion_log_probability: jax.Array
    locomotion_entropy: jax.Array
    locomotion_reward: jax.Array
    command_tracking_error: jax.Array
    done: jax.Array
    role: jax.Array
    learner_index: jax.Array
    episode_id: jax.Array
    environment_id: jax.Array
    root_position_world: jax.Array
    root_orientation_wxyz: jax.Array
    root_linear_velocity_world: jax.Array
    root_angular_velocity: jax.Array
    joint_position: jax.Array
    joint_velocity: jax.Array
    relative_position: jax.Array
    relative_velocity: jax.Array
    root_height: jax.Array
    up_alignment: jax.Array
    fallen: jax.Array
    near_fall: jax.Array
    out_of_bounds: jax.Array
    tag_contact: jax.Array
    tagged: jax.Array
    timed_out: jax.Array
    terrain_quaternion_wxyz: jax.Array


@dataclass(frozen=True)
class HierarchicalTrajectoryMetadata:
    opponent_id: str
    opponent_category: str
    learner_role: str
    opponent_role: str
    learner_index: int
    opponent_index: int


def trajectory_metadata(
    segment: HierarchicalRolloutSegment,
) -> HierarchicalTrajectoryMetadata:
    return HierarchicalTrajectoryMetadata(
        opponent_id=segment.opponent_id,
        opponent_category=segment.opponent_category,
        learner_role=segment.learner_role,
        opponent_role=segment.opponent_role,
        learner_index=segment.learner_index,
        opponent_index=segment.opponent_index,
    )


def locomotion_reward_terms(
    *,
    command: jax.Array,
    measured_linear_velocity: jax.Array,
    measured_yaw_rate: jax.Array,
    projected_gravity: jax.Array,
    joint_deviation: jax.Array,
    action: jax.Array,
    previous_action: jax.Array,
    fallen: jax.Array,
    config: LocomotionRewardConfig,
) -> LocomotionRewardTerms:
    tracking_error = command - jp.concatenate(
        (measured_linear_velocity, measured_yaw_rate[None])
    )
    linear_tracking = config.linear_tracking_weight * jp.exp(
        -jp.sum(jp.square(tracking_error[:2]))
    )
    yaw_tracking = config.yaw_tracking_weight * jp.exp(
        -jp.square(tracking_error[2])
    )
    upright = config.upright_weight * jp.exp(
        -jp.sum(jp.square(projected_gravity[:2]))
    )
    action_rate = -config.action_rate_weight * jp.sum(
        jp.square(action - previous_action)
    )
    joint_deviation_term = -config.joint_deviation_weight * jp.sum(
        jp.square(joint_deviation)
    )
    fall = -config.fall_penalty * fallen.astype(jp.float32)
    total = (
        linear_tracking
        + yaw_tracking
        + upright
        + action_rate
        + joint_deviation_term
        + fall
    )
    return LocomotionRewardTerms(
        linear_tracking=linear_tracking,
        yaw_tracking=yaw_tracking,
        upright=upright,
        action_rate=action_rate,
        joint_deviation=joint_deviation_term,
        fall=fall,
        total=total,
    )


def collect_hierarchical_transition(
    runner: HierarchicalRolloutRunner,
    segment: HierarchicalRolloutSegment,
    state: HierarchicalRolloutState,
    *,
    episode_id: jax.Array,
    environment_id: jax.Array,
    pursuer_config: TagPursuerConfig | None = None,
    evader_config: TagEvaderConfig | None = None,
    locomotion_config: LocomotionRewardConfig | None = None,
) -> tuple[HierarchicalRolloutState, HierarchicalTransition]:
    """Advance one control step and retain only learner training outputs."""
    pursuer_config = TagPursuerConfig() if pursuer_config is None else pursuer_config
    evader_config = TagEvaderConfig() if evader_config is None else evader_config
    locomotion_config = (
        LocomotionRewardConfig()
        if locomotion_config is None
        else locomotion_config
    )
    learner_index = segment.learner_index
    previous_strategy = strategy_observations(
        state.environment,
        state.commands,
        state.roles,
        runner.observation_config,
    ).as_array()[learner_index]
    previous_distance = jp.linalg.norm(
        state.environment.observation.relative_position[learner_index]
    )
    previous_action = state.previous_actions[learner_index]
    next_state = runner.step(segment, state)
    command = next_state.commands[learner_index]

    motor_fields = locomotion_observations(
        state.environment,
        runner.locomotion_layout,
        state.previous_actions,
        state.phases,
        runner.environment.default_joint_targets,
    )
    motor_observation = runner.locomotion_normalization.normalize(
        motor_fields.as_array(next_state.commands)
    )[learner_index]
    current_distance = jp.linalg.norm(
        next_state.environment.observation.relative_position[learner_index]
    )
    termination = next_state.environment.termination
    positions = next_state.environment.diagnostics.planar_position
    if segment.learner_is_pursuer:
        strategy_reward = pursuer_reward(
            previous_distance=previous_distance,
            current_distance=current_distance,
            previous_command=state.commands[learner_index],
            current_command=command,
            pursuer_planar_position=positions[learner_index],
            arena_half_extent=runner.environment.config.arena_half_extent,
            termination=termination,
            pursuer_index=learner_index,
            config=pursuer_config,
        ).total
    else:
        agent = runner.locomotion_layout.agents[learner_index]
        planar_velocity = next_state.environment.physics.data.qvel[
            agent.qvel_slice.start : agent.qvel_slice.start + 2
        ]
        strategy_reward = evader_reward(
            previous_distance=previous_distance,
            current_distance=current_distance,
            previous_command=state.commands[learner_index],
            current_command=command,
            evader_planar_position=positions[learner_index],
            evader_planar_velocity=planar_velocity,
            arena_half_extent=runner.environment.config.arena_half_extent,
            termination=termination,
            evader_index=learner_index,
            config=evader_config,
        ).total

    next_motor_fields = locomotion_observations(
        next_state.environment,
        runner.locomotion_layout,
        next_state.previous_actions,
        next_state.phases,
        runner.environment.default_joint_targets,
    )
    measured_command = jp.concatenate(
        (
            next_motor_fields.root_linear_velocity[learner_index, :2],
            next_motor_fields.root_angular_velocity[learner_index, 2:3],
        )
    )
    fallen = termination.fallen | next_state.environment.diagnostics.fall_detected
    locomotion_reward = locomotion_reward_terms(
        command=command,
        measured_linear_velocity=measured_command[:2],
        measured_yaw_rate=measured_command[2],
        projected_gravity=next_motor_fields.projected_gravity[learner_index],
        joint_deviation=next_motor_fields.joint_position[learner_index],
        action=next_state.previous_actions[learner_index],
        previous_action=previous_action,
        fallen=fallen[learner_index],
        config=locomotion_config,
    )

    data = next_state.environment.physics.data
    qpos = jp.stack(
        [data.qpos[agent.qpos_slice] for agent in runner.locomotion_layout.agents]
    )
    qvel = jp.stack(
        [data.qvel[agent.qvel_slice] for agent in runner.locomotion_layout.agents]
    )
    near_fall = ~fallen & (
        (next_state.environment.diagnostics.root_height < 0.60)
        | (next_state.environment.diagnostics.up_alignment < 0.80)
    )
    transition = HierarchicalTransition(
        strategy_observation=previous_strategy,
        strategy_command=command,
        strategy_value=next_state.strategy_values[learner_index],
        strategy_log_probability=next_state.strategy_log_probabilities[
            learner_index
        ],
        strategy_entropy=next_state.strategy_entropies[learner_index],
        strategy_reward=strategy_reward,
        strategy_update=(
            state.locomotion_steps
            % runner.config.locomotion_steps_per_strategy_step
            == 0
        ),
        locomotion_observation=motor_observation,
        locomotion_action=next_state.previous_actions[learner_index],
        locomotion_value=next_state.locomotion_values[learner_index],
        locomotion_log_probability=next_state.locomotion_log_probabilities[
            learner_index
        ],
        locomotion_entropy=next_state.locomotion_entropies[learner_index],
        locomotion_reward=locomotion_reward.total,
        command_tracking_error=command - measured_command,
        done=next_state.environment.done,
        role=state.roles[learner_index],
        learner_index=jp.asarray(learner_index, dtype=jp.int32),
        episode_id=jp.asarray(episode_id, dtype=jp.int32),
        environment_id=jp.asarray(environment_id, dtype=jp.int32),
        root_position_world=qpos[:, :3],
        root_orientation_wxyz=qpos[:, 3:7],
        root_linear_velocity_world=qvel[:, :3],
        root_angular_velocity=qvel[:, 3:6],
        joint_position=qpos[:, 7:],
        joint_velocity=qvel[:, 6:],
        relative_position=next_state.environment.observation.relative_position,
        relative_velocity=next_state.environment.observation.relative_velocity,
        root_height=next_state.environment.diagnostics.root_height,
        up_alignment=next_state.environment.diagnostics.up_alignment,
        fallen=fallen,
        near_fall=near_fall,
        out_of_bounds=(
            termination.out_of_bounds
            | next_state.environment.diagnostics.out_of_bounds
        ),
        tag_contact=next_state.environment.diagnostics.tag_contact,
        tagged=termination.tagged,
        timed_out=termination.timed_out,
        terrain_quaternion_wxyz=data.mocap_quat[
            runner.environment.model_bundle.terrain_mocap_id
        ],
    )
    return next_state, transition
