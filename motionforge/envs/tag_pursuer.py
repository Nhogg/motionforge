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
    boundary_margin: float = 1.0
    boundary_penalty_scale: float = 1.0
    tag_reward: float = 10.0
    pursuer_fall_penalty: float = 10.0
    pursuer_out_of_bounds_penalty: float = 10.0

    def __post_init__(self) -> None:
        if self.action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        if self.relative_velocity_scale <= 0.0:
            raise ValueError("relative_velocity_scale must be positive")
        if self.boundary_margin <= 0.0:
            raise ValueError("boundary_margin must be positive")
        for name in (
            "progress_reward_scale",
            "proximity_reward_scale",
            "step_penalty",
            "command_change_penalty_scale",
            "boundary_penalty_scale",
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
    boundary: jax.Array
    tag: jax.Array
    pursuer_fall: jax.Array
    pursuer_out_of_bounds: jax.Array
    total: jax.Array


@struct.dataclass
class TagPursuerPipelineState:
    """All physical and controller memory that must reset between episodes."""

    tag_state: object
    distance: jax.Array
    last_actions: jax.Array
    phases: jax.Array
    previous_command: jax.Array
    rng: jax.Array


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
    pursuer_planar_position: jax.Array,
    arena_half_extent: float,
    termination: TagEnvironmentTermination,
    pursuer_index: int,
    config: TagPursuerConfig,
) -> PursuerRewardTerms:
    """Compute independently logged high-level pursuer reward terms."""
    if pursuer_planar_position.shape != (2,):
        raise ValueError("pursuer_planar_position must have shape (2,)")
    if config.boundary_margin > arena_half_extent:
        raise ValueError("boundary_margin must not exceed arena_half_extent")
    progress = config.progress_reward_scale * (previous_distance - current_distance)
    proximity = config.proximity_reward_scale * jp.exp(-current_distance)
    step = jp.asarray(-config.step_penalty, dtype=jp.float32)
    command_change = -config.command_change_penalty_scale * jp.sum(
        jp.square(current_command - previous_command)
    )
    safe_half_extent = arena_half_extent - config.boundary_margin
    boundary_intrusion = jp.clip(
        (jp.max(jp.abs(pursuer_planar_position)) - safe_half_extent)
        / config.boundary_margin,
        0.0,
        1.0,
    )
    boundary = -config.boundary_penalty_scale * jp.square(boundary_intrusion)
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
        + boundary
        + tag
        + pursuer_fall
        + pursuer_out_of_bounds
    )
    return PursuerRewardTerms(
        progress=progress,
        proximity=proximity,
        step=step,
        command_change=command_change,
        boundary=boundary,
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
        if self.config.boundary_margin > self.tag_environment.config.arena_half_extent:
            raise ValueError("boundary_margin must not exceed arena_half_extent")
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
        pipeline_state = TagPursuerPipelineState(
            tag_state=tag_state,
            distance=distance,
            last_actions=jp.zeros((2, 29), dtype=jp.float32),
            phases=jp.asarray([[0.0, jp.pi], [0.0, jp.pi]]),
            previous_command=previous_command,
            rng=rollout_rng,
        )
        return State(
            pipeline_state=pipeline_state,
            obs=self._observation(tag_state, previous_command),
            reward=jp.zeros((), dtype=jp.float32),
            done=jp.zeros((), dtype=jp.float32),
            metrics=self._zero_metrics(),
            info={
                "time_out": jp.zeros((), dtype=jp.float32),
                "truncation": jp.zeros((), dtype=jp.float32),
            },
        )

    def _zero_metrics(self) -> dict[str, jax.Array]:
        return {
            name: jp.zeros((), dtype=jp.float32)
            for name in (
                "distance",
                "reward/command_change",
                "reward/boundary",
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
        pipeline = state.pipeline_state

        def locomotion_step(carry, _):
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

            return advance(carry), None

        tag_state, last_actions, phases, rng = jax.lax.scan(
            locomotion_step,
            (
                pipeline.tag_state,
                pipeline.last_actions,
                pipeline.phases,
                pipeline.rng,
            ),
            xs=None,
            length=self.config.action_repeat,
        )[0]
        distance = jp.linalg.norm(
            tag_state.observation.relative_position[self.pursuer_index]
        )
        reward = pursuer_reward(
            previous_distance=pipeline.distance,
            current_distance=distance,
            previous_command=pipeline.previous_command,
            current_command=command,
            pursuer_planar_position=tag_state.diagnostics.planar_position[
                self.pursuer_index
            ],
            arena_half_extent=self.tag_environment.config.arena_half_extent,
            termination=tag_state.termination,
            pursuer_index=self.pursuer_index,
            config=self.config,
        )
        metrics = dict(state.metrics)
        metrics.update(
            {
                "distance": distance,
                "reward/command_change": reward.command_change,
                "reward/boundary": reward.boundary,
                "reward/progress": reward.progress,
                "reward/proximity": reward.proximity,
                "reward/pursuer_fall": reward.pursuer_fall,
                "reward/pursuer_out_of_bounds": reward.pursuer_out_of_bounds,
                "reward/step": reward.step,
                "reward/tag": reward.tag,
                "tagged": tag_state.termination.tagged.astype(jp.float32),
            }
        )
        next_pipeline = pipeline.replace(
            tag_state=tag_state,
            distance=distance,
            last_actions=last_actions,
            phases=phases,
            previous_command=command,
            rng=rng,
        )
        info = dict(state.info)
        info["truncation"] = tag_state.termination.timed_out.astype(jp.float32)
        info["time_out"] = tag_state.termination.timed_out.astype(jp.float32)
        return state.replace(
            pipeline_state=next_pipeline,
            obs=self._observation(tag_state, command),
            reward=reward.total,
            done=tag_state.done.astype(jp.float32),
            metrics=metrics,
            info=info,
        )


class TagEpisodeWrapper(brax_training.Wrapper):
    """Brax episode accounting that preserves task-provided truncations."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int) -> None:
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["steps"] = jp.zeros(rng.shape[:-1])
        state.info["episode_done"] = jp.zeros(rng.shape[:-1])
        episode_metrics = {
            "sum_reward": jp.zeros(rng.shape[:-1]),
            "length": jp.zeros(rng.shape[:-1]),
        }
        episode_metrics.update(
            {name: jp.zeros(rng.shape[:-1]) for name in state.metrics}
        )
        state.info["episode_metrics"] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def repeated_step(current_state, _):
            next_state = self.env.step(current_state, action)
            return next_state, next_state.reward

        state, rewards = jax.lax.scan(
            repeated_step,
            state,
            xs=None,
            length=self.action_repeat,
        )
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info["steps"] + self.action_repeat
        horizon = jp.asarray(self.episode_length, dtype=jp.int32)
        horizon_reached = steps >= horizon
        task_done = state.done
        horizon_truncation = horizon_reached & ~task_done.astype(bool)
        truncation = jp.maximum(
            state.info["truncation"], horizon_truncation.astype(jp.float32)
        )
        done = jp.where(horizon_reached, jp.ones_like(task_done), task_done)
        state.info["truncation"] = truncation
        state.info["time_out"] = truncation
        state.info["steps"] = steps

        previous_done = state.info["episode_done"]
        state.info["episode_metrics"]["sum_reward"] *= 1 - previous_done
        state.info["episode_metrics"]["sum_reward"] += jp.sum(rewards, axis=0)
        state.info["episode_metrics"]["length"] *= 1 - previous_done
        state.info["episode_metrics"]["length"] += self.action_repeat
        for name in state.info["episode_metrics"]:
            if name in {"sum_reward", "length"}:
                continue
            state.info["episode_metrics"][name] *= 1 - previous_done
            state.info["episode_metrics"][name] += state.metrics[name]
        state.info["episode_done"] = done
        return state.replace(done=done)


class TagAutoResetWrapper(brax_training.Wrapper):
    """Auto-reset with fresh seeded states and MJX-Warp-safe data merging."""

    @staticmethod
    def _split_keys(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        split = jax.vmap(jax.random.split)(rng)
        return split[:, 0], split[:, 1]

    def reset(self, rng: jax.Array) -> State:
        reset_rng, autoreset_rng = self._split_keys(rng)
        state = self.env.reset(reset_rng)
        state.info["autoreset_rng"] = autoreset_rng
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            state.info["steps"] = jp.where(
                state.done, jp.zeros_like(state.info["steps"]), state.info["steps"]
            )
        autoreset_rng, reset_rng = self._split_keys(state.info["autoreset_rng"])
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        reset_state = self.env.reset(reset_rng)
        state.info["autoreset_rng"] = autoreset_rng
        reset_pipeline = reset_state.pipeline_state
        current_pipeline = state.pipeline_state
        done = state.done.astype(bool)

        def select(reset_value, current_value):
            mask = done.reshape(done.shape + (1,) * (current_value.ndim - done.ndim))
            return jp.where(mask, reset_value, current_value)

        reset_tag = reset_pipeline.tag_state
        current_tag = current_pipeline.tag_state
        tag_state = current_tag.replace(
            data=jax.vmap(
                lambda current, reset, terminal: current.where(terminal, reset)
            )(current_tag.data, reset_tag.data, done),
            observation=jax.tree.map(
                select, reset_tag.observation, current_tag.observation
            ),
            diagnostics=jax.tree.map(
                select, reset_tag.diagnostics, current_tag.diagnostics
            ),
            termination=jax.tree.map(
                select, reset_tag.termination, current_tag.termination
            ),
            rng=select(reset_tag.rng, current_tag.rng),
            step_count=select(reset_tag.step_count, current_tag.step_count),
            fall_counts=select(reset_tag.fall_counts, current_tag.fall_counts),
            done=select(reset_tag.done, current_tag.done),
        )
        pipeline_state = current_pipeline.replace(
            tag_state=tag_state,
            distance=select(reset_pipeline.distance, current_pipeline.distance),
            last_actions=select(
                reset_pipeline.last_actions, current_pipeline.last_actions
            ),
            phases=select(reset_pipeline.phases, current_pipeline.phases),
            previous_command=select(
                reset_pipeline.previous_command, current_pipeline.previous_command
            ),
            rng=select(reset_pipeline.rng, current_pipeline.rng),
        )
        obs_mask = done.reshape(done.shape + (1,) * (state.obs.ndim - done.ndim))
        obs = jp.where(obs_mask, reset_state.obs, state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


def wrap_tag_pursuer_for_training(
    env: Env,
    episode_length: int = 200,
    action_repeat: int = 1,
    randomization_fn=None,
) -> Env:
    """Vectorize and auto-reset while preserving timeout bootstrapping."""
    if randomization_fn is not None:
        raise ValueError("Tag pursuer training does not support domain randomization")
    wrapped = brax_training.VmapWrapper(env)
    wrapped = TagEpisodeWrapper(wrapped, episode_length, action_repeat)
    return TagAutoResetWrapper(wrapped)
