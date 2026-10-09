"""End-to-end iteration orchestration for hierarchical PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jp

from motionforge.rollout import (
    AdvantageConfig,
    AdvantageTargets,
    HierarchicalRolloutBatch,
    HierarchicalRolloutRunner,
    HierarchicalRolloutState,
    collect_hierarchical_rollout,
    hierarchical_advantages,
)
from motionforge.rollout.trajectory import HierarchicalTrajectoryMetadata
from motionforge.training.hierarchical_update import (
    HierarchicalPpoUpdater,
    HierarchicalTrainState,
    HierarchicalUpdateMetrics,
)


@dataclass(frozen=True)
class HierarchicalTrainerConfig:
    rollout_horizon: int = 100
    strategy_advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    locomotion_advantage: AdvantageConfig = field(default_factory=AdvantageConfig)

    def __post_init__(self) -> None:
        if self.rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive")


@dataclass(frozen=True)
class HierarchicalIterationResult:
    batch: HierarchicalRolloutBatch
    strategy_targets: AdvantageTargets
    locomotion_targets: AdvantageTargets
    update_metrics: HierarchicalUpdateMetrics
    trajectory: HierarchicalTrajectoryMetadata


class HierarchicalPpoTrainer:
    """Connect rollout collection, two-timescale GAE, and PPO updates."""

    def __init__(
        self,
        runner: HierarchicalRolloutRunner,
        updater: HierarchicalPpoUpdater,
        config: HierarchicalTrainerConfig | None = None,
    ) -> None:
        if not runner.config.stochastic_actions:
            raise ValueError("training requires stochastic rollout actions")
        self.runner = runner
        self.updater = updater
        self.config = HierarchicalTrainerConfig() if config is None else config
        cadence = runner.config.locomotion_steps_per_strategy_step
        if self.config.rollout_horizon % cadence != 0:
            raise ValueError("rollout_horizon must be divisible by strategy cadence")

    def train_iteration(
        self,
        train_state: HierarchicalTrainState,
        rollout_state: HierarchicalRolloutState,
        *,
        seed: int,
        episode_id: int,
        environment_id: int,
    ) -> tuple[
        HierarchicalTrainState,
        HierarchicalRolloutState,
        HierarchicalIterationResult,
    ]:
        """Collect one on-policy segment and apply one scheduled update."""
        segment = self.runner.start_segment(train_state.parameters, seed=seed)
        next_rollout_state, batch = collect_hierarchical_rollout(
            self.runner,
            segment,
            rollout_state,
            horizon=self.config.rollout_horizon,
            episode_id=jp.asarray(episode_id, dtype=jp.int32),
            environment_id=jp.asarray(environment_id, dtype=jp.int32),
        )
        learner_index = segment.learner_index
        not_terminal = ~next_rollout_state.environment.done
        strategy_values, locomotion_values = self.runner.bootstrap_values(
            segment, next_rollout_state
        )
        strategy_bootstrap = jp.where(
            not_terminal,
            strategy_values[learner_index],
            0.0,
        )
        locomotion_bootstrap = jp.where(
            not_terminal,
            locomotion_values[learner_index],
            0.0,
        )
        strategy_targets, locomotion_targets = hierarchical_advantages(
            batch,
            strategy_bootstrap_value=strategy_bootstrap,
            locomotion_bootstrap_value=locomotion_bootstrap,
            strategy_config=self.config.strategy_advantage,
            locomotion_config=self.config.locomotion_advantage,
        )
        next_train_state, update_metrics = self.updater.update(
            train_state,
            batch.strategy,
            strategy_targets,
            batch.locomotion,
            locomotion_targets,
        )
        result = HierarchicalIterationResult(
            batch=batch,
            strategy_targets=strategy_targets,
            locomotion_targets=locomotion_targets,
            update_metrics=update_metrics,
            trajectory=HierarchicalTrajectoryMetadata(
                opponent_id=segment.opponent_id,
                opponent_category=segment.opponent_category,
                learner_role=segment.learner_role,
                opponent_role=segment.opponent_role,
                learner_index=segment.learner_index,
                opponent_index=segment.opponent_index,
            ),
        )
        return next_train_state, next_rollout_state, result
