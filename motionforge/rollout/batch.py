"""Fixed-horizon collection and two-timescale advantage construction."""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jp
from flax import struct

from motionforge.envs.tag_endings import TagEndCause
from motionforge.rollout.hierarchical import (
    HierarchicalRolloutRunner,
    HierarchicalRolloutSegment,
    HierarchicalRolloutState,
)
from motionforge.rollout.tag_reward import tag_strategy_reward_terms
from motionforge.rollout.trajectory import (
    HierarchicalTransition,
    collect_hierarchical_transition,
)


@dataclass(frozen=True)
class AdvantageConfig:
    discount: float = 0.99
    gae_lambda: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")


@struct.dataclass
class AdvantageTargets:
    advantages: jax.Array
    returns: jax.Array


@struct.dataclass
class StrategyBatch:
    observation: jax.Array
    command: jax.Array
    value: jax.Array
    log_probability: jax.Array
    entropy: jax.Array
    reward: jax.Array
    done: jax.Array
    valid: jax.Array
    role: jax.Array
    episode_id: jax.Array
    environment_id: jax.Array
    terminated: jax.Array | None = None
    truncated: jax.Array | None = None


@struct.dataclass
class LocomotionBatch:
    observation: jax.Array
    action: jax.Array
    value: jax.Array
    log_probability: jax.Array
    entropy: jax.Array
    reward: jax.Array
    done: jax.Array
    valid: jax.Array
    command_tracking_error: jax.Array
    episode_id: jax.Array
    environment_id: jax.Array
    terminated: jax.Array | None = None
    truncated: jax.Array | None = None


@struct.dataclass
class HierarchicalRolloutBatch:
    transitions: HierarchicalTransition
    transition_valid: jax.Array
    strategy: StrategyBatch
    locomotion: LocomotionBatch


def generalized_advantage_estimate(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    valid: jax.Array,
    bootstrap_value: jax.Array,
    config: AdvantageConfig,
    *,
    truncated: jax.Array | None = None,
    next_values: jax.Array | None = None,
) -> AdvantageTargets:
    """GAE: dones means TRUE termination, not an episode/reset boundary.

    Truncation retains value bootstrap but stops the advantage trace. Explicit
    next_values are required for truncations followed by another episode;
    collected TAG segments contain one episode and use the final-state value.
    Invalid padding must not replace that final-state bootstrap with pad values.
    """
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if values.shape != rewards.shape:
        raise ValueError("values must match rewards")
    if dones.shape != rewards.shape or valid.shape != rewards.shape:
        raise ValueError("dones and valid must match rewards")
    truncated = jp.zeros_like(dones) if truncated is None else truncated
    if truncated.shape != rewards.shape:
        raise ValueError("truncated must match rewards")
    if next_values is not None and next_values.shape != rewards.shape:
        raise ValueError("next_values must match rewards")
    supplied_next = jp.zeros_like(values) if next_values is None else next_values

    def reverse_step(carry, inputs):
        next_advantage, next_value = carry
        reward, value, done, is_truncated, is_valid, supplied = inputs
        next_value = next_value if next_values is None else supplied
        continues = (~done).astype(value.dtype)
        delta = reward + config.discount * continues * next_value - value
        advantage = delta + (
            config.discount
            * config.gae_lambda
            * continues
            * (~is_truncated).astype(value.dtype)
            * next_advantage
        )
        advantage = jp.where(is_valid, advantage, 0.0)
        return (advantage, jp.where(is_valid, value, carry[1])), advantage

    (_, _), advantages_reversed = jax.lax.scan(
        reverse_step,
        (jp.zeros_like(bootstrap_value), bootstrap_value),
        (
            rewards[::-1],
            values[::-1],
            dones[::-1],
            truncated[::-1],
            valid[::-1],
            supplied_next[::-1],
        ),
    )
    advantages = advantages_reversed[::-1]
    return AdvantageTargets(
        advantages=advantages,
        returns=jp.where(valid, advantages + values, 0.0),
    )


def _strategy_batch(
    transitions: HierarchicalTransition,
    valid: jax.Array,
    steps_per_strategy: int,
) -> StrategyBatch:
    horizon = transitions.strategy_reward.shape[0]
    if horizon % steps_per_strategy != 0:
        raise ValueError("rollout horizon must be divisible by strategy cadence")
    groups = horizon // steps_per_strategy

    def grouped(value: jax.Array) -> jax.Array:
        return value.reshape((groups, steps_per_strategy) + value.shape[1:])

    grouped_valid = grouped(valid)
    grouped_done = grouped(transitions.done)
    rewards = grouped(transitions.strategy_reward)
    return StrategyBatch(
        observation=grouped(transitions.strategy_observation)[:, 0],
        command=grouped(transitions.strategy_command)[:, 0],
        value=grouped(transitions.strategy_value)[:, 0],
        log_probability=grouped(transitions.strategy_log_probability)[:, 0],
        entropy=grouped(transitions.strategy_entropy)[:, 0],
        reward=jp.sum(jp.where(grouped_valid, rewards, 0.0), axis=1),
        done=jp.any(grouped_done & grouped_valid, axis=1),
        valid=jp.any(grouped_valid, axis=1),
        role=grouped(transitions.role)[:, 0],
        episode_id=grouped(transitions.episode_id)[:, 0],
        environment_id=grouped(transitions.environment_id)[:, 0],
        terminated=jp.any(grouped(transitions.terminated) & grouped_valid, axis=1),
        truncated=jp.any(grouped(transitions.truncated) & grouped_valid, axis=1),
    )


def collect_hierarchical_rollout(
    runner: HierarchicalRolloutRunner,
    segment: HierarchicalRolloutSegment,
    initial_state: HierarchicalRolloutState,
    *,
    horizon: int,
    episode_id: jax.Array,
    environment_id: jax.Array,
    strategy_discount: float = 0.99,
) -> tuple[HierarchicalRolloutState, HierarchicalRolloutBatch]:
    """Collect one fixed-shape segment while masking post-terminal slots."""
    cadence = runner.config.locomotion_steps_per_strategy_step
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if horizon % cadence != 0:
        raise ValueError("horizon must be divisible by strategy cadence")

    def scan_step(carry, _):
        state, alive = carry
        next_state, transition = collect_hierarchical_transition(
            runner,
            segment,
            state,
            episode_id=episode_id,
            environment_id=environment_id,
        )
        valid = alive
        # Retain the actual final observation, not a post-cutoff simulated state.
        if hasattr(state.environment, "physics"):
            # Warp contact buffers are globally packed, not ordinary per-world
            # array leaves. A vmapped cond/tree-select adds an invalid world
            # dimension to them. Data.where is the supported native merge.
            def select(old, new):
                return jp.where(alive, new, old)

            old_env, new_env = state.environment, next_state.environment
            physics = new_env.physics.replace(
                data=new_env.physics.data.where(~alive, old_env.physics.data),
                step_count=select(
                    old_env.physics.step_count, new_env.physics.step_count
                ),
            )
            environment = new_env.replace(
                physics=physics,
                observation=jax.tree.map(
                    select, old_env.observation, new_env.observation
                ),
                diagnostics=jax.tree.map(
                    select, old_env.diagnostics, new_env.diagnostics
                ),
                termination=jax.tree.map(
                    select, old_env.termination, new_env.termination
                ),
                fall_counts=select(old_env.fall_counts, new_env.fall_counts),
                done=select(old_env.done, new_env.done),
            )
            next_state = next_state.replace(
                environment=environment,
                **{
                    item.name: select(
                        getattr(state, item.name), getattr(next_state, item.name)
                    )
                    for item in fields(state)
                    if item.name != "environment"
                },
            )
        else:
            next_state = jax.tree.map(
                lambda old, new: jp.where(alive, new, old), state, next_state
            )
        alive = alive & ~transition.done
        return (next_state, alive), (transition, valid)

    (final_state, _), (transitions, valid) = jax.lax.scan(
        scan_step,
        (initial_state, ~initial_state.environment.done),
        xs=None,
        length=horizon,
    )
    # A shorter collector horizon is also an artificial cutoff, even if the
    # environment's configured time limit has not yet fired.
    cutoff = ~final_state.environment.done & jp.any(valid)
    ending = final_state.environment.termination
    ending = ending.replace(
        truncated=ending.truncated | cutoff,
        timed_out=ending.timed_out | cutoff,
        cause=jp.where(cutoff, TagEndCause.TIME_LIMIT_TRUNCATION, ending.cause),
    )
    final_state = final_state.replace(
        environment=final_state.environment.replace(
            termination=ending, done=ending.terminated | ending.truncated
        )
    )
    transitions = transitions.replace(
        truncated=transitions.truncated.at[-1].set(transitions.truncated[-1] | cutoff),
        timed_out=transitions.timed_out.at[-1].set(transitions.timed_out[-1] | cutoff),
        done=transitions.done.at[-1].set(transitions.done[-1] | cutoff),
        terminal_cause=transitions.terminal_cause.at[-1].set(
            jp.where(
                cutoff,
                TagEndCause.TIME_LIMIT_TRUNCATION,
                transitions.terminal_cause[-1],
            )
        ),
    )

    # Potential differences are evaluated ONCE per 10 Hz strategy step. Store
    # every component at the interval's last valid motor transition, allowing
    # existing aggregation and episode masking to remain exact.
    groups = horizon // cadence
    grouped_valid = valid.reshape(groups, cadence)
    counts = jp.sum(grouped_valid, axis=1)
    first = jp.arange(groups) * cadence
    last = first + jp.maximum(counts - 1, 0)
    learner = segment.learner_index
    components = tag_strategy_reward_terms(
        previous_distance=transitions.previous_distance[first],
        current_distance=transitions.planar_distance[last],
        command=transitions.strategy_command[first],
        previous_command=transitions.previous_command[first],
        position=transitions.root_position_world[last, learner, :2],
        arena_half_extent=runner.environment.config.arena_half_extent,
        tagged=transitions.tagged[last],
        own_fall=transitions.persistent_fallen[last, learner],
        own_oob=transitions.out_of_bounds[last, learner],
        learner_is_pursuer=segment.learner_is_pursuer,
        config=runner.config.tag_reward,
        discount=strategy_discount,
    )
    # Simulator failures are explicitly classified, never rewarded as progress.
    components = jax.tree.map(
        lambda value: jp.where(
            (counts > 0) & (transitions.terminal_cause[last] != TagEndCause.NONFINITE),
            value,
            0.0,
        ),
        components,
    )
    component_transitions = jax.tree.map(
        lambda value: jp.zeros((horizon,), value.dtype).at[last].set(value), components
    )
    transitions = transitions.replace(
        strategy_reward=component_transitions.total,
        strategy_reward_components=component_transitions,
    )
    locomotion = LocomotionBatch(
        observation=transitions.locomotion_observation,
        action=transitions.locomotion_action,
        value=transitions.locomotion_value,
        log_probability=transitions.locomotion_log_probability,
        entropy=transitions.locomotion_entropy,
        reward=jp.where(valid, transitions.locomotion_reward, 0.0),
        done=transitions.done,
        valid=valid,
        command_tracking_error=transitions.command_tracking_error,
        episode_id=transitions.episode_id,
        environment_id=transitions.environment_id,
        terminated=transitions.terminated,
        truncated=transitions.truncated,
    )
    return final_state, HierarchicalRolloutBatch(
        transitions=transitions,
        transition_valid=valid,
        strategy=_strategy_batch(transitions, valid, cadence),
        locomotion=locomotion,
    )


def hierarchical_advantages(
    batch: HierarchicalRolloutBatch,
    *,
    strategy_bootstrap_value: jax.Array,
    locomotion_bootstrap_value: jax.Array,
    strategy_config: AdvantageConfig | None = None,
    locomotion_config: AdvantageConfig | None = None,
) -> tuple[AdvantageTargets, AdvantageTargets]:
    """Build independent strategy and locomotion GAE targets."""
    strategy_config = AdvantageConfig() if strategy_config is None else strategy_config
    locomotion_config = (
        AdvantageConfig() if locomotion_config is None else locomotion_config
    )
    strategy = generalized_advantage_estimate(
        batch.strategy.reward,
        batch.strategy.value,
        batch.strategy.done
        if batch.strategy.terminated is None
        else batch.strategy.terminated,
        batch.strategy.valid,
        strategy_bootstrap_value,
        strategy_config,
        truncated=batch.strategy.truncated,
    )
    locomotion = generalized_advantage_estimate(
        batch.locomotion.reward,
        batch.locomotion.value,
        batch.locomotion.done
        if batch.locomotion.terminated is None
        else batch.locomotion.terminated,
        batch.locomotion.valid,
        locomotion_bootstrap_value,
        locomotion_config,
        truncated=batch.locomotion.truncated,
    )
    return strategy, locomotion
