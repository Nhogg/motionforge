"""Fixed-horizon collection and two-timescale advantage construction."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp
from flax import struct

from motionforge.rollout.hierarchical import (
    HierarchicalRolloutRunner,
    HierarchicalRolloutSegment,
    HierarchicalRolloutState,
)
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
    reward: jax.Array
    done: jax.Array
    valid: jax.Array
    role: jax.Array
    episode_id: jax.Array
    environment_id: jax.Array


@struct.dataclass
class LocomotionBatch:
    observation: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    valid: jax.Array
    command_tracking_error: jax.Array
    episode_id: jax.Array
    environment_id: jax.Array


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
) -> AdvantageTargets:
    """Compute masked GAE for one temporal level in forward-time order."""
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if values.shape != rewards.shape:
        raise ValueError("values must match rewards")
    if dones.shape != rewards.shape or valid.shape != rewards.shape:
        raise ValueError("dones and valid must match rewards")

    def reverse_step(carry, inputs):
        next_advantage, next_value = carry
        reward, value, done, is_valid = inputs
        continues = (~done).astype(value.dtype)
        delta = reward + config.discount * continues * next_value - value
        advantage = delta + (
            config.discount
            * config.gae_lambda
            * continues
            * next_advantage
        )
        advantage = jp.where(is_valid, advantage, 0.0)
        return (advantage, value), advantage

    (_, _), advantages_reversed = jax.lax.scan(
        reverse_step,
        (jp.zeros_like(bootstrap_value), bootstrap_value),
        (rewards[::-1], values[::-1], dones[::-1], valid[::-1]),
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
        reward=jp.sum(jp.where(grouped_valid, rewards, 0.0), axis=1),
        done=jp.any(grouped_done & grouped_valid, axis=1),
        valid=jp.any(grouped_valid, axis=1),
        role=grouped(transitions.role)[:, 0],
        episode_id=grouped(transitions.episode_id)[:, 0],
        environment_id=grouped(transitions.environment_id)[:, 0],
    )


def collect_hierarchical_rollout(
    runner: HierarchicalRolloutRunner,
    segment: HierarchicalRolloutSegment,
    initial_state: HierarchicalRolloutState,
    *,
    horizon: int,
    episode_id: jax.Array,
    environment_id: jax.Array,
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
        alive = alive & ~transition.done
        return (next_state, alive), (transition, valid)

    (final_state, _), (transitions, valid) = jax.lax.scan(
        scan_step,
        (initial_state, jp.asarray(True)),
        xs=None,
        length=horizon,
    )
    locomotion = LocomotionBatch(
        observation=transitions.locomotion_observation,
        action=transitions.locomotion_action,
        value=transitions.locomotion_value,
        reward=jp.where(valid, transitions.locomotion_reward, 0.0),
        done=transitions.done,
        valid=valid,
        command_tracking_error=transitions.command_tracking_error,
        episode_id=transitions.episode_id,
        environment_id=transitions.environment_id,
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
        batch.strategy.done,
        batch.strategy.valid,
        strategy_bootstrap_value,
        strategy_config,
    )
    locomotion = generalized_advantage_estimate(
        batch.locomotion.reward,
        batch.locomotion.value,
        batch.locomotion.done,
        batch.locomotion.valid,
        locomotion_bootstrap_value,
        locomotion_config,
    )
    return strategy, locomotion
