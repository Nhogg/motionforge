"""End-to-end iteration orchestration for hierarchical PPO training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
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


def validate_full_episode_rollout(
    *,
    rollout_horizon: int,
    episode_duration: float,
    control_timestep: float,
) -> None:
    """Require one training rollout to expose the task's terminal outcomes."""
    if rollout_horizon <= 0:
        raise ValueError("rollout_horizon must be positive")
    if episode_duration <= 0.0:
        raise ValueError("episode_duration must be positive")
    if control_timestep <= 0.0:
        raise ValueError("control_timestep must be positive")
    rollout_duration = rollout_horizon * control_timestep
    if not math.isclose(rollout_duration, episode_duration, abs_tol=1e-9):
        raise ValueError(
            "rollout_horizon must span exactly one episode at the configured "
            "control timestep"
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

    def train_batch_iteration(
        self,
        train_state: HierarchicalTrainState,
        rollout_states: HierarchicalRolloutState,
        *,
        seed: int,
        episode_id: int,
    ) -> tuple[
        HierarchicalTrainState,
        HierarchicalRolloutState,
        HierarchicalIterationResult,
    ]:
        """Collect and flatten one vectorized environment batch before updating."""
        environment_count = rollout_states.commands.shape[0]
        if environment_count <= 0:
            raise ValueError("rollout_states must contain at least one environment")
        if environment_count % 2 != 0:
            raise ValueError("balanced role batches require an even environment count")
        pursuer_segment = self.runner.start_segment(
            train_state.parameters,
            seed=seed,
            learner_is_pursuer=True,
        )
        evader_segment = self.runner.start_segment(
            train_state.parameters,
            seed=seed,
            learner_is_pursuer=False,
        )
        environment_ids = jp.arange(environment_count, dtype=jp.int32)
        episode_ids = jp.full((environment_count,), episode_id, dtype=jp.int32)
        split = environment_count // 2

        collect = jax.vmap(
            lambda segment_state, current_episode_id, environment_id, segment: (
                collect_hierarchical_rollout(
                    self.runner,
                    segment,
                    segment_state,
                    horizon=self.config.rollout_horizon,
                    episode_id=current_episode_id,
                    environment_id=environment_id,
                )
            ),
            in_axes=(0, 0, 0, None),
        )
        pursuer_states, pursuer_batches = collect(
            rollout_states,
            episode_ids,
            environment_ids,
            pursuer_segment,
        )
        evader_states, evader_batches = collect(
            rollout_states,
            episode_ids,
            environment_ids,
            evader_segment,
        )
        pursuer_mask = environment_ids < split

        def select_array(pursuer_value, evader_value):
            mask = pursuer_mask.reshape(
                pursuer_mask.shape + (1,) * (pursuer_value.ndim - pursuer_mask.ndim)
            )
            return jp.where(mask, pursuer_value, evader_value)

        environment = evader_states.environment.replace(
            physics=evader_states.environment.physics.replace(
                data=jax.vmap(
                    lambda evader, pursuer, use_pursuer: evader.where(
                        use_pursuer, pursuer
                    )
                )(
                    evader_states.environment.physics.data,
                    pursuer_states.environment.physics.data,
                    pursuer_mask,
                ),
                step_count=select_array(
                    pursuer_states.environment.physics.step_count,
                    evader_states.environment.physics.step_count,
                ),
            ),
            observation=jax.tree.map(
                select_array,
                pursuer_states.environment.observation,
                evader_states.environment.observation,
            ),
            diagnostics=jax.tree.map(
                select_array,
                pursuer_states.environment.diagnostics,
                evader_states.environment.diagnostics,
            ),
            termination=jax.tree.map(
                select_array,
                pursuer_states.environment.termination,
                evader_states.environment.termination,
            ),
            fall_counts=select_array(
                pursuer_states.environment.fall_counts,
                evader_states.environment.fall_counts,
            ),
            done=select_array(
                pursuer_states.environment.done,
                evader_states.environment.done,
            ),
        )
        next_rollout_states = evader_states.replace(
            environment=environment,
            **{
                field: select_array(
                    getattr(pursuer_states, field), getattr(evader_states, field)
                )
                for field in (
                    "strategy_actions",
                    "commands",
                    "previous_actions",
                    "phases",
                    "roles",
                    "strategy_values",
                    "locomotion_values",
                    "strategy_log_probabilities",
                    "strategy_entropies",
                    "locomotion_log_probabilities",
                    "locomotion_entropies",
                    "policy_rng",
                    "locomotion_steps",
                    "strategy_updates",
                )
            },
        )
        environment_batches = jax.tree.map(
            select_array, pursuer_batches, evader_batches
        )

        pursuer_values = jax.vmap(
            lambda state: self.runner.bootstrap_values(pursuer_segment, state)
        )(pursuer_states)
        evader_values = jax.vmap(
            lambda state: self.runner.bootstrap_values(evader_segment, state)
        )(evader_states)
        strategy_values = select_array(pursuer_values[0], evader_values[0])
        locomotion_values = select_array(pursuer_values[1], evader_values[1])
        learner_indices = jp.where(
            environment_ids < split,
            pursuer_segment.learner_index,
            evader_segment.learner_index,
        )
        batch_indices = jp.arange(environment_count)
        not_terminal = ~next_rollout_states.environment.done
        strategy_bootstrap = jp.where(
            not_terminal,
            strategy_values[batch_indices, learner_indices],
            0.0,
        )
        locomotion_bootstrap = jp.where(
            not_terminal,
            locomotion_values[batch_indices, learner_indices],
            0.0,
        )
        strategy_targets, locomotion_targets = jax.vmap(
            lambda batch, strategy_value, locomotion_value: hierarchical_advantages(
                batch,
                strategy_bootstrap_value=strategy_value,
                locomotion_bootstrap_value=locomotion_value,
                strategy_config=self.config.strategy_advantage,
                locomotion_config=self.config.locomotion_advantage,
            )
        )(
            environment_batches,
            strategy_bootstrap,
            locomotion_bootstrap,
        )

        def flatten_environment_and_time(value: jax.Array) -> jax.Array:
            return value.reshape((-1,) + value.shape[2:])

        batch = jax.tree.map(flatten_environment_and_time, environment_batches)
        strategy_targets = jax.tree.map(flatten_environment_and_time, strategy_targets)
        locomotion_targets = jax.tree.map(
            flatten_environment_and_time, locomotion_targets
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
                opponent_id=pursuer_segment.opponent_id,
                opponent_category=pursuer_segment.opponent_category,
                learner_role="balanced",
                opponent_role="balanced",
                learner_index=-1,
                opponent_index=-1,
            ),
        )
        return next_train_state, next_rollout_states, result
