"""Single-agent high-level evader environment for P7 PPO training.

The learned evader issues bounded velocity commands while the accepted P7
pursuer and P2 locomotion controller remain deterministic and frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jp
from brax.envs.base import Env, State
from brax.envs.wrappers import training as brax_training
from flax import struct

from motionforge.controllers import (
    G1LocomotionController,
    build_g1_tag_policy_observation_layout,
    g1_tag_policy_observation,
)
from motionforge.envs.g1_standing import G1StandingJoystick, default_config
from motionforge.envs.tag_environment import (
    TagEnvironmentConfig,
    TagEnvironmentTermination,
    TwoG1TagEnvironment,
)
from motionforge.envs.tag_pursuer import (
    TagAutoResetWrapper,
    TagEpisodeWrapper,
    pursuer_action_to_command,
    pursuer_observation,
)
from motionforge.envs.tag_roles import TagRole
from motionforge.policies import FrozenLearnedPursuer


@dataclass(frozen=True)
class TagEvaderConfig:
    """High-level evader observation, action, and reward contract."""

    action_repeat: int = 5
    relative_velocity_scale: float = 2.0
    separation_reward_scale: float = 2.0
    proximity_penalty_scale: float = 0.02
    survival_reward: float = 0.002
    command_change_penalty_scale: float = 0.01
    boundary_margin: float = 1.0
    boundary_penalty_scale: float = 1.0
    boundary_outward_velocity_penalty_scale: float = 5.0
    timeout_reward: float = 10.0
    tag_penalty: float = 10.0
    evader_fall_penalty: float = 25.0
    evader_out_of_bounds_penalty: float = 25.0

    def __post_init__(self) -> None:
        if self.action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        if self.relative_velocity_scale <= 0.0:
            raise ValueError("relative_velocity_scale must be positive")
        if self.boundary_margin <= 0.0:
            raise ValueError("boundary_margin must be positive")
        for name in (
            "separation_reward_scale",
            "proximity_penalty_scale",
            "survival_reward",
            "command_change_penalty_scale",
            "boundary_penalty_scale",
            "boundary_outward_velocity_penalty_scale",
            "timeout_reward",
            "tag_penalty",
            "evader_fall_penalty",
            "evader_out_of_bounds_penalty",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")


@struct.dataclass
class EvaderRewardTerms:
    separation: jax.Array
    proximity: jax.Array
    survival: jax.Array
    command_change: jax.Array
    boundary: jax.Array
    boundary_outward_velocity: jax.Array
    timeout: jax.Array
    tag: jax.Array
    evader_fall: jax.Array
    evader_out_of_bounds: jax.Array
    total: jax.Array


@struct.dataclass
class TagEvaderPipelineState:
    """Physical state plus learned and frozen high-level controller memory."""

    tag_state: object
    distance: jax.Array
    last_actions: jax.Array
    phases: jax.Array
    previous_command: jax.Array
    pursuer_previous_command: jax.Array
    rng: jax.Array


def evader_reward(
    *,
    previous_distance: jax.Array,
    current_distance: jax.Array,
    previous_command: jax.Array,
    current_command: jax.Array,
    evader_planar_position: jax.Array,
    evader_planar_velocity: jax.Array,
    arena_half_extent: float,
    termination: TagEnvironmentTermination,
    evader_index: int,
    config: TagEvaderConfig,
) -> EvaderRewardTerms:
    """Compute independently logged high-level evader reward terms."""
    if evader_planar_position.shape != (2,):
        raise ValueError("evader_planar_position must have shape (2,)")
    if evader_planar_velocity.shape != (2,):
        raise ValueError("evader_planar_velocity must have shape (2,)")
    if config.boundary_margin > arena_half_extent:
        raise ValueError("boundary_margin must not exceed arena_half_extent")
    separation = config.separation_reward_scale * (current_distance - previous_distance)
    proximity = -config.proximity_penalty_scale * jp.exp(-current_distance)
    survival = jp.asarray(config.survival_reward, dtype=jp.float32)
    command_change = -config.command_change_penalty_scale * jp.sum(
        jp.square(current_command - previous_command)
    )
    safe_half_extent = arena_half_extent - config.boundary_margin
    intrusion = jp.clip(
        (jp.max(jp.abs(evader_planar_position)) - safe_half_extent)
        / config.boundary_margin,
        0.0,
        1.0,
    )
    boundary = -config.boundary_penalty_scale * jp.square(intrusion)
    axis_intrusion = jp.clip(
        (jp.abs(evader_planar_position) - safe_half_extent)
        / config.boundary_margin,
        0.0,
        1.0,
    )
    outward_velocity = jp.maximum(
        jp.sign(evader_planar_position) * evader_planar_velocity,
        0.0,
    )
    boundary_outward_velocity = (
        -config.boundary_outward_velocity_penalty_scale
        * jp.sum(axis_intrusion * outward_velocity)
    )
    timeout = config.timeout_reward * termination.timed_out.astype(jp.float32)
    tag = -config.tag_penalty * termination.tagged.astype(jp.float32)
    evader_fall = -config.evader_fall_penalty * termination.fallen[
        evader_index
    ].astype(jp.float32)
    evader_out_of_bounds = (
        -config.evader_out_of_bounds_penalty
        * termination.out_of_bounds[evader_index].astype(jp.float32)
    )
    total = (
        separation
        + proximity
        + survival
        + command_change
        + boundary
        + boundary_outward_velocity
        + timeout
        + tag
        + evader_fall
        + evader_out_of_bounds
    )
    return EvaderRewardTerms(
        separation=separation,
        proximity=proximity,
        survival=survival,
        command_change=command_change,
        boundary=boundary,
        boundary_outward_velocity=boundary_outward_velocity,
        timeout=timeout,
        tag=tag,
        evader_fall=evader_fall,
        evader_out_of_bounds=evader_out_of_bounds,
        total=total,
    )


class TagEvaderEnvironment(Env):
    """Brax environment that learns only the evader's velocity command."""

    def __init__(
        self,
        *,
        locomotion_checkpoint: Path,
        pursuer_checkpoint: Path,
        pursuer_fixed_noise_std: float = 0.2,
        tag_config: TagEnvironmentConfig | None = None,
        evader_config: TagEvaderConfig | None = None,
    ) -> None:
        self.tag_environment = TwoG1TagEnvironment(tag_config)
        self.config = evader_config or TagEvaderConfig()
        if self.config.boundary_margin > self.tag_environment.config.arena_half_extent:
            raise ValueError("boundary_margin must not exceed arena_half_extent")
        pursuer_index = self.tag_environment.roles.by_role(TagRole.PURSUER).index
        self.evader_index = self.tag_environment.roles.by_role(TagRole.EVADER).index
        self.pursuer = FrozenLearnedPursuer(
            agent_index=pursuer_index,
            checkpoint_path=pursuer_checkpoint,
            fixed_noise_std=pursuer_fixed_noise_std,
        )

        source_config = default_config()
        source_config.impl = "warp"
        source_config.naconmax = 16
        source_config.njmax = 128
        source_config.push_config.enable = False
        source_environment = G1StandingJoystick(config=source_config)
        self.controller = G1LocomotionController.from_checkpoint(
            locomotion_checkpoint, source_environment, deterministic=True
        )
        self.policy_observation_layout = build_g1_tag_policy_observation_layout(
            self.tag_environment.model_bundle
        )
        self.default_pose = jp.asarray(source_environment._default_pose)
        self.phase_delta = (
            2.0 * jp.pi * self.tag_environment.config.control_timestep * 1.375
        )

    @property
    def observation_size(self) -> int:
        return 9

    @property
    def action_size(self) -> int:
        return 3

    @property
    def backend(self) -> str:
        return "warp"

    def _agent_observation(self, tag_state, agent_index, previous_command):
        return pursuer_observation(
            tag_state.observation,
            agent_index,
            previous_command,
            self.tag_environment.config.arena_half_extent,
            self.config.relative_velocity_scale,
        )

    def _zero_metrics(self) -> dict[str, jax.Array]:
        return {
            name: jp.zeros((), dtype=jp.float32)
            for name in (
                "distance",
                "reward/boundary",
                "reward/boundary_outward_velocity",
                "reward/command_change",
                "reward/evader_fall",
                "reward/evader_out_of_bounds",
                "reward/proximity",
                "reward/separation",
                "reward/survival",
                "reward/tag",
                "reward/timeout",
                "tagged",
                "timed_out",
            )
        }

    def reset(self, rng: jax.Array) -> State:
        reset_rng, rollout_rng = jax.random.split(rng)
        tag_state = self.tag_environment.reset(reset_rng)
        zero_command = jp.zeros(3, dtype=jp.float32)
        distance = jp.linalg.norm(
            tag_state.observation.relative_position[self.evader_index]
        )
        pipeline = TagEvaderPipelineState(
            tag_state=tag_state,
            distance=distance,
            last_actions=jp.zeros((2, 29), dtype=jp.float32),
            phases=jp.asarray([[0.0, jp.pi], [0.0, jp.pi]]),
            previous_command=zero_command,
            pursuer_previous_command=zero_command,
            rng=rollout_rng,
        )
        return State(
            pipeline_state=pipeline,
            obs=self._agent_observation(tag_state, self.evader_index, zero_command),
            reward=jp.zeros((), dtype=jp.float32),
            done=jp.zeros((), dtype=jp.float32),
            metrics=self._zero_metrics(),
            info={
                "time_out": jp.zeros((), dtype=jp.float32),
                "truncation": jp.zeros((), dtype=jp.float32),
            },
        )

    def step(self, state: State, action: jax.Array) -> State:
        evader_command = pursuer_action_to_command(action)
        pipeline = state.pipeline_state
        pursuer_observation_value = self._agent_observation(
            pipeline.tag_state,
            self.pursuer.agent_index,
            pipeline.pursuer_previous_command,
        )
        pursuer_command = self.pursuer.command(pursuer_observation_value)

        def locomotion_step(carry, _):
            active_state, active_actions, active_phases, active_rng = carry
            commands = jp.zeros((2, 3), dtype=jp.float32)
            commands = commands.at[self.pursuer.agent_index].set(pursuer_command)
            commands = commands.at[self.evader_index].set(evader_command)
            active_rng, agent0_rng, agent1_rng = jax.random.split(active_rng, 3)
            observations = tuple(
                g1_tag_policy_observation(
                    active_state.data,
                    self.policy_observation_layout,
                    agent_index,
                    commands[agent_index],
                    active_actions[agent_index],
                    active_phases[agent_index],
                    self.default_pose,
                )
                for agent_index in range(2)
            )
            outputs = (
                self.controller.act(observations[0], agent0_rng),
                self.controller.act(observations[1], agent1_rng),
            )
            next_actions = jp.stack([output.normalized_action for output in outputs])
            joint_targets = jp.stack(
                [output.joint_position_targets for output in outputs]
            )
            next_state = self.tag_environment.step(active_state, joint_targets)
            next_phases = (
                jp.fmod(active_phases + self.phase_delta + jp.pi, 2.0 * jp.pi)
                - jp.pi
            )
            return (next_state, next_actions, next_phases, active_rng), None

        tag_state, last_actions, phases, rng = jax.lax.scan(
            locomotion_step,
            (pipeline.tag_state, pipeline.last_actions, pipeline.phases, pipeline.rng),
            xs=None,
            length=self.config.action_repeat,
        )[0]
        distance = jp.linalg.norm(
            tag_state.observation.relative_position[self.evader_index]
        )
        reward = evader_reward(
            previous_distance=pipeline.distance,
            current_distance=distance,
            previous_command=pipeline.previous_command,
            current_command=evader_command,
            evader_planar_position=tag_state.diagnostics.planar_position[
                self.evader_index
            ],
            evader_planar_velocity=tag_state.data.sensordata[
                self.tag_environment.observation_layout.agents[
                    self.evader_index
                ].global_velocity_sensor_slice
            ][:2],
            arena_half_extent=self.tag_environment.config.arena_half_extent,
            termination=tag_state.termination,
            evader_index=self.evader_index,
            config=self.config,
        )
        metrics = dict(state.metrics)
        metrics.update(
            {
                "distance": distance,
                "reward/boundary": reward.boundary,
                "reward/boundary_outward_velocity": (
                    reward.boundary_outward_velocity
                ),
                "reward/command_change": reward.command_change,
                "reward/evader_fall": reward.evader_fall,
                "reward/evader_out_of_bounds": reward.evader_out_of_bounds,
                "reward/proximity": reward.proximity,
                "reward/separation": reward.separation,
                "reward/survival": reward.survival,
                "reward/tag": reward.tag,
                "reward/timeout": reward.timeout,
                "tagged": tag_state.termination.tagged.astype(jp.float32),
                "timed_out": tag_state.termination.timed_out.astype(jp.float32),
            }
        )
        next_pipeline = pipeline.replace(
            tag_state=tag_state,
            distance=distance,
            last_actions=last_actions,
            phases=phases,
            previous_command=evader_command,
            pursuer_previous_command=pursuer_command,
            rng=rng,
        )
        # Reaching the time limit is the evader's task success, not a truncation.
        info = dict(state.info)
        info["truncation"] = jp.zeros((), dtype=jp.float32)
        info["time_out"] = tag_state.termination.timed_out.astype(jp.float32)
        return state.replace(
            pipeline_state=next_pipeline,
            obs=self._agent_observation(tag_state, self.evader_index, evader_command),
            reward=reward.total,
            done=tag_state.done.astype(jp.float32),
            metrics=metrics,
            info=info,
        )


def wrap_tag_evader_for_training(
    env: Env,
    episode_length: int = 200,
    action_repeat: int = 1,
    randomization_fn=None,
) -> Env:
    """Vectorize and fresh-reset the single-agent evader environment."""
    if randomization_fn is not None:
        raise ValueError("Tag evader training does not support domain randomization")
    wrapped = brax_training.VmapWrapper(env)
    wrapped = TagEpisodeWrapper(wrapped, episode_length, action_repeat)
    return TagAutoResetWrapper(wrapped)
