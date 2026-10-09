"""Independent clipped-PPO objectives for both hierarchy levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jp
from flax import struct

from motionforge.models import (
    HierarchicalPolicyConfig,
    LocomotionModule,
    StrategyModule,
    diagonal_normal_entropy,
    tanh_normal_log_prob,
)
from motionforge.rollout import AdvantageTargets, LocomotionBatch, StrategyBatch


@dataclass(frozen=True)
class PpoLossConfig:
    clipping_epsilon: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if self.clipping_epsilon <= 0.0:
            raise ValueError("clipping_epsilon must be positive")
        if self.value_loss_coefficient < 0.0:
            raise ValueError("value_loss_coefficient must be nonnegative")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be nonnegative")


@struct.dataclass
class PpoLossMetrics:
    total_loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approximate_kl: jax.Array
    clip_fraction: jax.Array


def _masked_mean(value: jax.Array, valid: jax.Array) -> jax.Array:
    weights = valid.astype(value.dtype)
    return jp.sum(value * weights) / jp.maximum(jp.sum(weights), 1.0)


def _normalized_advantages(
    advantages: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    mean = _masked_mean(advantages, valid)
    variance = _masked_mean(jp.square(advantages - mean), valid)
    return jp.where(valid, (advantages - mean) / jp.sqrt(variance + 1e-8), 0.0)


def _clipped_ppo_loss(
    *,
    new_log_probability: jax.Array,
    new_value: jax.Array,
    entropy: jax.Array,
    old_log_probability: jax.Array,
    old_value: jax.Array,
    targets: AdvantageTargets,
    valid: jax.Array,
    config: PpoLossConfig,
) -> tuple[jax.Array, PpoLossMetrics]:
    advantages = targets.advantages
    if config.normalize_advantages:
        advantages = _normalized_advantages(advantages, valid)
    log_ratio = new_log_probability - old_log_probability
    ratio = jp.exp(log_ratio)
    clipped_ratio = jp.clip(
        ratio,
        1.0 - config.clipping_epsilon,
        1.0 + config.clipping_epsilon,
    )
    policy_loss = -_masked_mean(
        jp.minimum(ratio * advantages, clipped_ratio * advantages), valid
    )
    clipped_value = old_value + jp.clip(
        new_value - old_value,
        -config.clipping_epsilon,
        config.clipping_epsilon,
    )
    value_error = jp.square(new_value - targets.returns)
    clipped_value_error = jp.square(clipped_value - targets.returns)
    value_loss = 0.5 * _masked_mean(
        jp.maximum(value_error, clipped_value_error), valid
    )
    mean_entropy = _masked_mean(entropy, valid)
    approximate_kl = 0.5 * _masked_mean(jp.square(log_ratio), valid)
    clip_fraction = _masked_mean(
        (jp.abs(ratio - 1.0) > config.clipping_epsilon).astype(jp.float32),
        valid,
    )
    total = (
        policy_loss
        + config.value_loss_coefficient * value_loss
        - config.entropy_coefficient * mean_entropy
    )
    metrics = PpoLossMetrics(
        total_loss=total,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=mean_entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )
    return total, metrics


def strategy_ppo_loss(
    strategy_parameters: Any,
    batch: StrategyBatch,
    targets: AdvantageTargets,
    *,
    policy_config: HierarchicalPolicyConfig | None = None,
    loss_config: PpoLossConfig | None = None,
) -> tuple[jax.Array, PpoLossMetrics]:
    """Evaluate PPO using only the trainable strategy parameter subtree."""
    policy_config = (
        HierarchicalPolicyConfig() if policy_config is None else policy_config
    )
    loss_config = PpoLossConfig() if loss_config is None else loss_config
    output = StrategyModule(
        policy_config.strategy_hidden_layer_sizes,
        policy_config.command_scale,
        policy_config.strategy_initial_log_std,
    ).apply({"params": strategy_parameters}, batch.observation)
    scale = jp.broadcast_to(
        jp.asarray(policy_config.command_scale), output.location.shape
    )
    log_probability = tanh_normal_log_prob(
        batch.command,
        output.location,
        output.log_std,
        scale,
    )
    return _clipped_ppo_loss(
        new_log_probability=log_probability,
        new_value=output.value,
        entropy=diagonal_normal_entropy(output.log_std),
        old_log_probability=batch.log_probability,
        old_value=batch.value,
        targets=targets,
        valid=batch.valid,
        config=loss_config,
    )


def locomotion_ppo_loss(
    locomotion_parameters: Any,
    batch: LocomotionBatch,
    targets: AdvantageTargets,
    *,
    policy_config: HierarchicalPolicyConfig | None = None,
    loss_config: PpoLossConfig | None = None,
) -> tuple[jax.Array, PpoLossMetrics]:
    """Evaluate PPO using only the trainable locomotion parameter subtree."""
    policy_config = (
        HierarchicalPolicyConfig() if policy_config is None else policy_config
    )
    loss_config = PpoLossConfig() if loss_config is None else loss_config
    output = LocomotionModule(
        policy_config.locomotion_hidden_layer_sizes,
        policy_config.locomotion_initial_log_std,
    ).apply({"params": locomotion_parameters}, batch.observation)
    log_probability = tanh_normal_log_prob(
        batch.action,
        output.location,
        output.log_std,
        jp.ones_like(output.location),
    )
    return _clipped_ppo_loss(
        new_log_probability=log_probability,
        new_value=output.value,
        entropy=diagonal_normal_entropy(output.log_std),
        old_log_probability=batch.log_probability,
        old_value=batch.value,
        targets=targets,
        valid=batch.valid,
        config=loss_config,
    )
