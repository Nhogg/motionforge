"""Explicit two-optimizer update scheduling for hierarchical PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jp
import optax
from flax import struct
from flax.core import freeze, unfreeze

from motionforge.models import HierarchicalPolicyConfig
from motionforge.rollout import AdvantageTargets, LocomotionBatch, StrategyBatch
from motionforge.training.hierarchical_ppo import (
    PpoLossConfig,
    PpoLossMetrics,
    locomotion_ppo_loss,
    strategy_ppo_loss,
)


@dataclass(frozen=True)
class HierarchicalUpdateConfig:
    strategy_learning_rate: float = 3e-4
    locomotion_learning_rate: float = 1e-4
    strategy_update_interval: int = 1
    locomotion_update_interval: int = 1
    maximum_gradient_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.strategy_learning_rate <= 0.0:
            raise ValueError("strategy_learning_rate must be positive")
        if self.locomotion_learning_rate <= 0.0:
            raise ValueError("locomotion_learning_rate must be positive")
        if self.strategy_update_interval <= 0:
            raise ValueError("strategy_update_interval must be positive")
        if self.locomotion_update_interval <= 0:
            raise ValueError("locomotion_update_interval must be positive")
        if self.maximum_gradient_norm <= 0.0:
            raise ValueError("maximum_gradient_norm must be positive")


@struct.dataclass
class HierarchicalTrainState:
    parameters: Any
    strategy_optimizer_state: optax.OptState
    locomotion_optimizer_state: optax.OptState
    iteration: jax.Array
    strategy_updates: jax.Array
    locomotion_updates: jax.Array


@struct.dataclass
class HierarchicalUpdateMetrics:
    strategy: PpoLossMetrics
    locomotion: PpoLossMetrics
    strategy_gradient_norm: jax.Array
    locomotion_gradient_norm: jax.Array
    strategy_updated: jax.Array
    locomotion_updated: jax.Array


class HierarchicalPpoUpdater:
    """Apply logically and structurally separate PPO optimizer updates."""

    def __init__(
        self,
        config: HierarchicalUpdateConfig | None = None,
        *,
        policy_config: HierarchicalPolicyConfig | None = None,
        strategy_loss_config: PpoLossConfig | None = None,
        locomotion_loss_config: PpoLossConfig | None = None,
    ) -> None:
        self.config = HierarchicalUpdateConfig() if config is None else config
        self.policy_config = (
            HierarchicalPolicyConfig() if policy_config is None else policy_config
        )
        self.strategy_loss_config = (
            PpoLossConfig() if strategy_loss_config is None else strategy_loss_config
        )
        self.locomotion_loss_config = (
            PpoLossConfig()
            if locomotion_loss_config is None
            else locomotion_loss_config
        )
        self.strategy_optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.maximum_gradient_norm),
            optax.adam(self.config.strategy_learning_rate),
        )
        self.locomotion_optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.maximum_gradient_norm),
            optax.adam(self.config.locomotion_learning_rate),
        )

    def initialize(self, parameters: Any) -> HierarchicalTrainState:
        """Create independent optimizer state for each named parameter subtree."""
        if set(parameters) != {"strategy", "locomotion"}:
            raise ValueError("parameters must contain strategy and locomotion groups")
        parameters = freeze(unfreeze(parameters))
        return HierarchicalTrainState(
            parameters=parameters,
            strategy_optimizer_state=self.strategy_optimizer.init(
                parameters["strategy"]
            ),
            locomotion_optimizer_state=self.locomotion_optimizer.init(
                parameters["locomotion"]
            ),
            iteration=jp.zeros((), dtype=jp.int32),
            strategy_updates=jp.zeros((), dtype=jp.int32),
            locomotion_updates=jp.zeros((), dtype=jp.int32),
        )

    def update(
        self,
        state: HierarchicalTrainState,
        strategy_batch: StrategyBatch,
        strategy_targets: AdvantageTargets,
        locomotion_batch: LocomotionBatch,
        locomotion_targets: AdvantageTargets,
    ) -> tuple[HierarchicalTrainState, HierarchicalUpdateMetrics]:
        """Run whichever level updates are scheduled for this iteration."""
        strategy_due = state.iteration % self.config.strategy_update_interval == 0
        locomotion_due = state.iteration % self.config.locomotion_update_interval == 0

        def strategy_update(_):
            (loss, metrics), gradients = jax.value_and_grad(
                strategy_ppo_loss, has_aux=True
            )(
                state.parameters["strategy"],
                strategy_batch,
                strategy_targets,
                policy_config=self.policy_config,
                loss_config=self.strategy_loss_config,
            )
            del loss
            updates, optimizer_state = self.strategy_optimizer.update(
                gradients,
                state.strategy_optimizer_state,
                state.parameters["strategy"],
            )
            parameters = optax.apply_updates(state.parameters["strategy"], updates)
            return parameters, optimizer_state, metrics, optax.global_norm(gradients)

        def strategy_skip(_):
            _, metrics = strategy_ppo_loss(
                state.parameters["strategy"],
                strategy_batch,
                strategy_targets,
                policy_config=self.policy_config,
                loss_config=self.strategy_loss_config,
            )
            return (
                state.parameters["strategy"],
                state.strategy_optimizer_state,
                metrics,
                jp.zeros((), dtype=jp.float32),
            )

        def locomotion_update(_):
            (loss, metrics), gradients = jax.value_and_grad(
                locomotion_ppo_loss, has_aux=True
            )(
                state.parameters["locomotion"],
                locomotion_batch,
                locomotion_targets,
                policy_config=self.policy_config,
                loss_config=self.locomotion_loss_config,
            )
            del loss
            updates, optimizer_state = self.locomotion_optimizer.update(
                gradients,
                state.locomotion_optimizer_state,
                state.parameters["locomotion"],
            )
            parameters = optax.apply_updates(state.parameters["locomotion"], updates)
            return parameters, optimizer_state, metrics, optax.global_norm(gradients)

        def locomotion_skip(_):
            _, metrics = locomotion_ppo_loss(
                state.parameters["locomotion"],
                locomotion_batch,
                locomotion_targets,
                policy_config=self.policy_config,
                loss_config=self.locomotion_loss_config,
            )
            return (
                state.parameters["locomotion"],
                state.locomotion_optimizer_state,
                metrics,
                jp.zeros((), dtype=jp.float32),
            )

        (
            strategy_parameters,
            strategy_optimizer_state,
            strategy_metrics,
            strategy_norm,
        ) = jax.lax.cond(strategy_due, strategy_update, strategy_skip, operand=None)
        (
            locomotion_parameters,
            locomotion_optimizer_state,
            locomotion_metrics,
            locomotion_norm,
        ) = jax.lax.cond(
            locomotion_due, locomotion_update, locomotion_skip, operand=None
        )
        parameters = freeze(
            {
                "strategy": strategy_parameters,
                "locomotion": locomotion_parameters,
            }
        )
        next_state = state.replace(
            parameters=parameters,
            strategy_optimizer_state=strategy_optimizer_state,
            locomotion_optimizer_state=locomotion_optimizer_state,
            iteration=state.iteration + 1,
            strategy_updates=(state.strategy_updates + strategy_due.astype(jp.int32)),
            locomotion_updates=(
                state.locomotion_updates + locomotion_due.astype(jp.int32)
            ),
        )
        return next_state, HierarchicalUpdateMetrics(
            strategy=strategy_metrics,
            locomotion=locomotion_metrics,
            strategy_gradient_norm=strategy_norm,
            locomotion_gradient_norm=locomotion_norm,
            strategy_updated=strategy_due,
            locomotion_updated=locomotion_due,
        )
