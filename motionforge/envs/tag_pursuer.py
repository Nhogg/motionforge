"""Single-agent high-level pursuer environment for P7 PPO training.

The learned action is a bounded G1 velocity command. A frozen scripted evader
and frozen P2 locomotion controller supply the opponent command and both
robots' joint targets while the existing two-G1 environment owns physics and
tag rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jp
from brax.envs.base import Env, State
from flax import struct

from motionforge.controllers import (
    G1LocomotionController,
    build_g1_tag_policy_observation_layout,
    g1_tag_policy_observation,
)
from motionforge.envs.g1_standing import G1StandingJoystick, default_config
from motionforge.envs.tag_environment import (
    TagEnvironmentConfig,
    TagEnvironmentObservation,
    TagEnvironmentTermination,
    TwoG1TagEnvironment,
)
from motionforge.envs.tag_roles import TagRole
from motionforge.policies import FrozenScriptedEvader


@dataclass(frozen=True)
class TagPursuerConfig:
    """High-level observation, action, and reward contract."""

    action_repeat: int = 5
    relative_velocity_scale: float = 2.0
    progress_reward_scale: float = 2.0
    proximity_reward_scale: float = 0.02
    step_penalty: float = 0.002
    command_change_penalty_scale: float = 0.01
    tag_reward: float = 10.0
    pursuer_fall_penalty: float = 10.0
    pursuer_out_of_bounds_penalty: float = 10.0

    def __post_init__(self) -> None:
        if self.action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        if self.relative_velocity_scale <= 0.0:
            raise ValueError("relative_velocity_scale must be positive")
        for name in (
            "progress_reward_scale",
            "proximity_reward_scale",
            "step_penalty",
            "command_change_penalty_scale",
            "tag_reward",
            "pursuer_fall_penalty",
            "pursuer_out_of_bounds_penalty",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")


@struct.dataclass
class PursuerRewardTerms:
    progress: jax.Array
    proximity: jax.Array
    step: jax.Array
    command_change: jax.Array
    tag: jax.Array
    pursuer_fall: jax.Array
    pursuer_out_of_bounds: jax.Array
    total: jax.Array


def pursuer_action_to_command(action: jax.Array) -> jax.Array:
    """Map normalized PPO action to the accepted G1 command limits."""
    if action.shape != (3,):
        raise ValueError(f"action must have shape (3,), got {action.shape}")
    return jp.clip(action, -1.0, 1.0) * jp.asarray([1.0, 0.5, 1.0])


def pursuer_observation(
    observation: TagEnvironmentObservation,
    pursuer_index: int,
    previous_command: jax.Array,
    arena_half_extent: float,
    relative_velocity_scale: float,
) -> jax.Array:
    """Build the normalized 9-value non-clock pursuer observation."""
    if pursuer_index not in (0, 1):
        raise ValueError("pursuer_index must be 0 or 1")
    if previous_command.shape != (3,):
        raise ValueError("previous_command must have shape (3,)")
    if arena_half_extent <= 0.0 or relative_velocity_scale <= 0.0:
        raise ValueError("normalization scales must be positive")
    return jp.concatenate(
        [
            observation.relative_position[pursuer_index] / (2.0 * arena_half_extent),
            observation.relative_velocity[pursuer_index] / relative_velocity_scale,
            observation.arena_center_position[pursuer_index] / arena_half_extent,
            previous_command / jp.asarray([1.0, 0.5, 1.0]),
        ]
    )


def pursuer_reward(
    *,
    previous_distance: jax.Array,
    current_distance: jax.Array,
    previous_command: jax.Array,
    current_command: jax.Array,
    termination: TagEnvironmentTermination,
    pursuer_index: int,
    config: TagPursuerConfig,
) -> PursuerRewardTerms:
    """Compute independently logged high-level pursuer reward terms."""
    progress = config.progress_reward_scale * (previous_distance - current_distance)
    proximity = config.proximity_reward_scale * jp.exp(-current_distance)
    step = jp.asarray(-config.step_penalty, dtype=jp.float32)
    command_change = -config.command_change_penalty_scale * jp.sum(
        jp.square(current_command - previous_command)
    )
    tag = config.tag_reward * termination.tagged.astype(jp.float32)
    pursuer_fall = -config.pursuer_fall_penalty * termination.fallen[
        pursuer_index
    ].astype(jp.float32)
    pursuer_out_of_bounds = (
        -config.pursuer_out_of_bounds_penalty
        * termination.out_of_bounds[pursuer_index].astype(jp.float32)
    )
    total = (
        progress
        + proximity
        + step
        + command_change
        + tag
        + pursuer_fall
        + pursuer_out_of_bounds
    )
    return PursuerRewardTerms(
        progress=progress,
        proximity=proximity,
        step=step,
        command_change=command_change,
        tag=tag,
        pursuer_fall=pursuer_fall,
        pursuer_out_of_bounds=pursuer_out_of_bounds,
        total=total,
    )


class TagPursuerEnvironment(Env):
    """Brax environment that learns only the pursuer's velocity command."""

    def __init__(
        self,
        *,
        locomotion_checkpoint: Path,
        tag_config: TagEnvironmentConfig | None = None,
        pursuer_config: TagPursuerConfig | None = None,
    ) -> None:
        self.tag_environment = TwoG1TagEnvironment(tag_config)
        self.config = pursuer_config or TagPursuerConfig()
        self.pursuer_index = self.tag_environment.roles.by_role(TagRole.PURSUER).index
        evader_index = self.tag_environment.roles.by_role(TagRole.EVADER).index
        self.evader = FrozenScriptedEvader(agent_index=evader_index)

        source_config = default_config()
        source_config.impl = "warp"
        source_config.naconmax = 16
        source_config.njmax = 128
        source_config.push_config.enable = False
        source_environment = G1StandingJoystick(config=source_config)
        self.controller = G1LocomotionController.from_checkpoint(
            locomotion_checkpoint,
            source_environment,
            deterministic=True,
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

    def _observation(self, tag_state, previous_command):
        return pursuer_observation(
            tag_state.observation,
            self.pursuer_index,
            previous_command,
            self.tag_environment.config.arena_half_extent,
            self.config.relative_velocity_scale,
        )

    def reset(self, rng: jax.Array) -> State:
        reset_rng, rollout_rng = jax.random.split(rng)
        tag_state = self.tag_environment.reset(reset_rng)
        previous_command = jp.zeros(3, dtype=jp.float32)
        distance = jp.linalg.norm(
            tag_state.observation.relative_position[self.pursuer_index]
        )
        return State(
            pipeline_state=tag_state,
            obs=self._observation(tag_state, previous_command),
            reward=jp.zeros((), dtype=jp.float32),
            done=jp.zeros((), dtype=jp.float32),
            metrics=self._zero_metrics(),
            info={
                "distance": distance,
                "last_actions": jp.zeros((2, 29), dtype=jp.float32),
                "phases": jp.asarray([[0.0, jp.pi], [0.0, jp.pi]]),
                "previous_command": previous_command,
                "rng": rollout_rng,
                "truncation": jp.zeros((), dtype=jp.float32),
            },
        )

    def _zero_metrics(self) -> dict[str, jax.Array]:
        return {
            name: jp.zeros((), dtype=jp.float32)
            for name in (
                "distance",
                "reward/command_change",
                "reward/progress",
                "reward/proximity",
                "reward/pursuer_fall",
                "reward/pursuer_out_of_bounds",
                "reward/step",
                "reward/tag",
                "tagged",
            )
        }

    def step(self, state: State, action: jax.Array) -> State:
        command = pursuer_action_to_command(action)

        def locomotion_step(carry, _):
            tag_state = carry[0]

            def advance(active_carry):
                active_state, active_actions, active_phases, active_rng = active_carry
                evader_command = self.evader.command(active_state.observation)
                commands = jp.zeros((2, 3), dtype=jp.float32)
                commands = commands.at[self.pursuer_index].set(command)
                commands = commands.at[self.evader.agent_index].set(evader_command)
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
                next_actions = jp.stack(
                    [output.normalized_action for output in outputs]
                )
                joint_targets = jp.stack(
                    [output.joint_position_targets for output in outputs]
                )
                next_state = self.tag_environment.step(active_state, joint_targets)
                next_phases = (
                    jp.fmod(active_phases + self.phase_delta + jp.pi, 2.0 * jp.pi)
                    - jp.pi
                )
                return next_state, next_actions, next_phases, active_rng

            next_carry = jax.lax.cond(
                tag_state.done,
                lambda unchanged: unchanged,
                advance,
                carry,
            )
            return next_carry, None

        tag_state, last_actions, phases, rng = jax.lax.scan(
            locomotion_step,
            (
                state.pipeline_state,
                state.info["last_actions"],
                state.info["phases"],
                state.info["rng"],
            ),
            xs=None,
            length=self.config.action_repeat,
        )[0]
        distance = jp.linalg.norm(
            tag_state.observation.relative_position[self.pursuer_index]
        )
        reward = pursuer_reward(
            previous_distance=state.info["distance"],
            current_distance=distance,
            previous_command=state.info["previous_command"],
            current_command=command,
            termination=tag_state.termination,
            pursuer_index=self.pursuer_index,
            config=self.config,
        )
        metrics = {
            "distance": distance,
            "reward/command_change": reward.command_change,
            "reward/progress": reward.progress,
            "reward/proximity": reward.proximity,
            "reward/pursuer_fall": reward.pursuer_fall,
            "reward/pursuer_out_of_bounds": reward.pursuer_out_of_bounds,
            "reward/step": reward.step,
            "reward/tag": reward.tag,
            "tagged": tag_state.termination.tagged.astype(jp.float32),
        }
        return state.replace(
            pipeline_state=tag_state,
            obs=self._observation(tag_state, command),
            reward=reward.total,
            done=tag_state.done.astype(jp.float32),
            metrics=metrics,
            info={
                "distance": distance,
                "last_actions": last_actions,
                "phases": phases,
                "previous_command": command,
                "rng": rng,
                "truncation": tag_state.termination.timed_out.astype(jp.float32),
            },
        )
