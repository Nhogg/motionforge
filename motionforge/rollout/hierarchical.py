"""Multirate execution for the role-conditioned hierarchical policy."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jp
from flax import struct
from flax.core import freeze, unfreeze

from motionforge.controllers import build_g1_tag_policy_observation_layout
from motionforge.envs import (
    HierarchicalObservationConfig,
    MultiAgentTaskEnv,
    MultiAgentTaskState,
    locomotion_observations,
    role_encodings,
    strategy_observations,
)
from motionforge.league import LeagueOpponent
from motionforge.models import (
    HierarchicalPolicy,
    LocomotionModule,
    LocomotionNormalization,
    StrategyModule,
    StrategyOutput,
    diagonal_normal_entropy,
    tanh_normal_sample_and_log_prob,
)


def sample_learner_is_pursuer(*, seed: int, probability: float) -> bool:
    """Sample a segment role reproducibly without requiring device state."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return random.Random(seed).random() < probability


@dataclass(frozen=True)
class HierarchicalRolloutConfig:
    physics_hz: int = 500
    locomotion_hz: int = 50
    strategy_hz: int = 10
    gait_frequency_hz: float = 1.375
    action_scale: float = 0.5
    learner_pursuer_probability: float = 0.5
    stochastic_actions: bool = False

    def __post_init__(self) -> None:
        for name in ("physics_hz", "locomotion_hz", "strategy_hz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.physics_hz % self.locomotion_hz != 0:
            raise ValueError("physics_hz must be divisible by locomotion_hz")
        if self.locomotion_hz % self.strategy_hz != 0:
            raise ValueError("locomotion_hz must be divisible by strategy_hz")
        if self.gait_frequency_hz <= 0.0:
            raise ValueError("gait_frequency_hz must be positive")
        if self.action_scale <= 0.0:
            raise ValueError("action_scale must be positive")
        if not 0.0 <= self.learner_pursuer_probability <= 1.0:
            raise ValueError("learner_pursuer_probability must be in [0, 1]")

    @property
    def locomotion_steps_per_strategy_step(self) -> int:
        return self.locomotion_hz // self.strategy_hz

    @property
    def physics_steps_per_locomotion_step(self) -> int:
        return self.physics_hz // self.locomotion_hz


@struct.dataclass
class HierarchicalRolloutState:
    environment: MultiAgentTaskState
    commands: jax.Array
    previous_actions: jax.Array
    phases: jax.Array
    roles: jax.Array
    strategy_values: jax.Array
    locomotion_values: jax.Array
    strategy_log_probabilities: jax.Array
    strategy_entropies: jax.Array
    locomotion_log_probabilities: jax.Array
    locomotion_entropies: jax.Array
    policy_rng: jax.Array
    locomotion_steps: jax.Array
    strategy_updates: jax.Array


@struct.dataclass
class HierarchicalRolloutSegment:
    """Parameters and league provenance fixed for one collection segment."""

    learner_parameters: Any
    opponent_parameters: Any
    opponent_id: str = struct.field(pytree_node=False)
    opponent_category: str = struct.field(pytree_node=False)
    learner_index: int = struct.field(pytree_node=False)
    learner_is_pursuer: bool = struct.field(pytree_node=False)

    def __post_init__(self) -> None:
        if self.learner_index not in (0, 1):
            raise ValueError("learner_index must be zero or one")

    @property
    def opponent_index(self) -> int:
        return 1 - self.learner_index

    @property
    def learner_role(self) -> str:
        return "pursuer" if self.learner_is_pursuer else "evader"

    @property
    def opponent_role(self) -> str:
        return "evader" if self.learner_is_pursuer else "pursuer"


class HierarchicalRolloutRunner:
    """Execute learner and frozen-opponent hierarchies at fixed temporal rates."""

    def __init__(
        self,
        environment: MultiAgentTaskEnv,
        locomotion_normalization: LocomotionNormalization,
        config: HierarchicalRolloutConfig | None = None,
    ) -> None:
        self.environment = environment
        self.locomotion_normalization = locomotion_normalization
        self.config = HierarchicalRolloutConfig() if config is None else config
        self.policy = HierarchicalPolicy()
        self.locomotion_layout = build_g1_tag_policy_observation_layout(
            environment.model_bundle
        )
        self.observation_config = HierarchicalObservationConfig(
            arena_half_extent=environment.config.arena_half_extent,
            command_scale=self.policy.config.command_scale,
        )

        simulation_hz = round(1.0 / environment.config.simulation_timestep)
        control_hz = round(1.0 / environment.config.control_timestep)
        if not math.isclose(
            simulation_hz,
            1.0 / environment.config.simulation_timestep,
            abs_tol=1e-9,
        ):
            raise ValueError("environment simulation timestep is not integral Hz")
        if not math.isclose(
            control_hz,
            1.0 / environment.config.control_timestep,
            abs_tol=1e-9,
        ):
            raise ValueError("environment control timestep is not integral Hz")
        if simulation_hz != self.config.physics_hz:
            raise ValueError("runner physics_hz does not match WarpEnv")
        if control_hz != self.config.locomotion_hz:
            raise ValueError("runner locomotion_hz does not match WarpEnv")
        if (
            environment.config.physics_substeps
            != self.config.physics_steps_per_locomotion_step
        ):
            raise ValueError("runner physics ratio does not match WarpEnv substeps")

        self.phase_delta = (
            2.0
            * jp.pi
            * environment.config.control_timestep
            * self.config.gait_frequency_hz
        )

    def reset(self, key: jax.Array) -> HierarchicalRolloutState:
        environment_key, policy_key = jax.random.split(key)
        environment_state = self.environment.reset(environment_key)
        return HierarchicalRolloutState(
            environment=environment_state,
            commands=jp.zeros((2, 3), dtype=jp.float32),
            previous_actions=jp.zeros((2, 29), dtype=jp.float32),
            phases=jp.asarray([[0.0, jp.pi], [0.0, jp.pi]], dtype=jp.float32),
            roles=role_encodings(jp.asarray(self.environment.config.pursuer_index)),
            strategy_values=jp.zeros((2,), dtype=jp.float32),
            locomotion_values=jp.zeros((2,), dtype=jp.float32),
            strategy_log_probabilities=jp.zeros((2,), dtype=jp.float32),
            strategy_entropies=jp.zeros((2,), dtype=jp.float32),
            locomotion_log_probabilities=jp.zeros((2,), dtype=jp.float32),
            locomotion_entropies=jp.zeros((2,), dtype=jp.float32),
            policy_rng=policy_key,
            locomotion_steps=jp.zeros((), dtype=jp.int32),
            strategy_updates=jp.zeros((), dtype=jp.int32),
        )

    def initialize_parameters(
        self,
        key: jax.Array,
        state: HierarchicalRolloutState,
    ):
        """Initialize both parameter groups from one representative state."""
        strategy = strategy_observations(
            state.environment,
            state.commands,
            state.roles,
            self.observation_config,
        )
        locomotion = locomotion_observations(
            state.environment,
            self.locomotion_layout,
            state.previous_actions,
            state.phases,
            self.environment.default_joint_targets,
        )
        return self.policy.init(
            key,
            strategy,
            locomotion,
            self.locomotion_normalization,
        )["params"]

    def bootstrap_values(
        self,
        segment: HierarchicalRolloutSegment,
        state: HierarchicalRolloutState,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate both value heads at the current, unstepped observation."""
        agent_parameters = self._agent_parameters(segment)
        strategy_observation = strategy_observations(
            state.environment,
            state.commands,
            state.roles,
            self.observation_config,
        ).as_array()
        strategy_module = StrategyModule(
            self.policy.config.strategy_hidden_layer_sizes,
            self.policy.config.command_scale,
            self.policy.config.strategy_initial_log_std,
        )
        strategy_values = jax.vmap(
            lambda parameters, observation: (
                strategy_module.apply({"params": parameters}, observation).value
            )
        )(agent_parameters["strategy"], strategy_observation)

        locomotion_observation = locomotion_observations(
            state.environment,
            self.locomotion_layout,
            state.previous_actions,
            state.phases,
            self.environment.default_joint_targets,
        ).as_array(state.commands)
        normalized_locomotion = self.locomotion_normalization.normalize(
            locomotion_observation
        )
        locomotion_module = LocomotionModule(
            self.policy.config.locomotion_hidden_layer_sizes,
            self.policy.config.locomotion_initial_log_std,
        )
        locomotion_values = jax.vmap(
            lambda parameters, observation: (
                locomotion_module.apply({"params": parameters}, observation).value
            )
        )(agent_parameters["locomotion"], normalized_locomotion)
        return strategy_values, locomotion_values

    def start_segment(
        self,
        learner_parameters: Any,
        *,
        seed: int,
    ) -> HierarchicalRolloutSegment:
        """Select and freeze the two policy trees used for one rollout segment."""
        learner_is_pursuer = self.sample_learner_is_pursuer(seed=seed)
        learner_index = (
            self.environment.config.pursuer_index
            if learner_is_pursuer
            else 1 - self.environment.config.pursuer_index
        )
        opponent = self.environment.sample_opponent(seed=seed)
        self._validate_opponent(opponent)
        if jax.tree.structure(learner_parameters) != jax.tree.structure(
            opponent.policy
        ):
            raise ValueError("learner and opponent parameter structures must match")
        learner_shapes = jax.tree.map(jp.shape, learner_parameters)
        opponent_shapes = jax.tree.map(jp.shape, opponent.policy)
        if jax.tree.leaves(learner_shapes) != jax.tree.leaves(opponent_shapes):
            raise ValueError("learner and opponent parameter shapes must match")
        if learner_index not in (0, 1):
            raise ValueError("learner_index must be zero or one")
        return HierarchicalRolloutSegment(
            learner_parameters=freeze(unfreeze(learner_parameters)),
            opponent_parameters=freeze(unfreeze(opponent.policy)),
            opponent_id=opponent.opponent_id,
            opponent_category=opponent.category,
            learner_index=learner_index,
            learner_is_pursuer=learner_is_pursuer,
        )

    def sample_learner_is_pursuer(self, *, seed: int) -> bool:
        """Sample the learner role deterministically at segment granularity."""
        return sample_learner_is_pursuer(
            seed=seed,
            probability=self.config.learner_pursuer_probability,
        )

    @staticmethod
    def _validate_opponent(opponent: LeagueOpponent) -> None:
        try:
            strategy = opponent.policy["strategy"]
            locomotion = opponent.policy["locomotion"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "league opponent policy must contain strategy and locomotion parameters"
            ) from error
        if not strategy or not locomotion:
            raise ValueError("league opponent parameter groups must not be empty")

    @staticmethod
    def _agent_parameters(segment: HierarchicalRolloutSegment) -> Any:
        """Stack parameter trees in physical agent order for vmapped inference."""
        ordered = [segment.opponent_parameters, segment.opponent_parameters]
        ordered[segment.learner_index] = segment.learner_parameters
        return jax.tree.map(lambda *leaves: jp.stack(leaves), *ordered)

    def step(
        self,
        segment: HierarchicalRolloutSegment,
        state: HierarchicalRolloutState,
    ) -> HierarchicalRolloutState:
        """Advance exactly one locomotion update and its ten physics steps."""
        strategy_observation = strategy_observations(
            state.environment,
            state.commands,
            state.roles,
            self.observation_config,
        )
        strategy_module = StrategyModule(
            self.policy.config.strategy_hidden_layer_sizes,
            self.policy.config.command_scale,
            self.policy.config.strategy_initial_log_std,
        )
        agent_parameters = self._agent_parameters(segment)
        next_policy_key, strategy_key, locomotion_key = jax.random.split(
            state.policy_rng, 3
        )
        strategy_keys = jax.random.split(strategy_key, 2)
        locomotion_keys = jax.random.split(locomotion_key, 2)
        should_update_strategy = (
            state.locomotion_steps % self.config.locomotion_steps_per_strategy_step == 0
        )

        def update_strategy(_):
            output = jax.vmap(
                lambda parameters, observation: strategy_module.apply(
                    {"params": parameters}, observation
                )
            )(
                agent_parameters["strategy"],
                strategy_observation.as_array(),
            )
            entropy = diagonal_normal_entropy(output.log_std)
            if not self.config.stochastic_actions:
                return output, jp.zeros((2,), dtype=jp.float32), entropy
            command, log_probability = jax.vmap(tanh_normal_sample_and_log_prob)(
                strategy_keys,
                output.location,
                output.log_std,
                jp.broadcast_to(
                    jp.asarray(self.policy.config.command_scale),
                    output.location.shape,
                ),
            )
            return output.replace(command=command), log_probability, entropy

        def hold_strategy(_):
            return (
                StrategyOutput(
                    location=jp.zeros_like(state.commands),
                    command=state.commands,
                    log_std=jp.zeros_like(state.commands),
                    value=state.strategy_values,
                ),
                state.strategy_log_probabilities,
                state.strategy_entropies,
            )

        strategy, strategy_log_probability, strategy_entropy = jax.lax.cond(
            should_update_strategy,
            update_strategy,
            hold_strategy,
            operand=None,
        )
        motor_commands = jax.lax.stop_gradient(strategy.command)
        locomotion_observation = locomotion_observations(
            state.environment,
            self.locomotion_layout,
            state.previous_actions,
            state.phases,
            self.environment.default_joint_targets,
        )
        normalized_locomotion = self.locomotion_normalization.normalize(
            locomotion_observation.as_array(motor_commands)
        )
        locomotion_module = LocomotionModule(
            self.policy.config.locomotion_hidden_layer_sizes,
            self.policy.config.locomotion_initial_log_std,
        )
        locomotion = jax.vmap(
            lambda parameters, observation: locomotion_module.apply(
                {"params": parameters}, observation
            )
        )(
            agent_parameters["locomotion"],
            normalized_locomotion,
        )
        locomotion_entropy = diagonal_normal_entropy(locomotion.log_std)
        if self.config.stochastic_actions:
            action, locomotion_log_probability = jax.vmap(
                tanh_normal_sample_and_log_prob
            )(
                locomotion_keys,
                locomotion.location,
                locomotion.log_std,
                jp.ones_like(locomotion.location),
            )
            locomotion = locomotion.replace(action=action)
        else:
            locomotion_log_probability = jp.zeros((2,), dtype=jp.float32)
        joint_targets = (
            self.environment.default_joint_targets
            + locomotion.action * self.config.action_scale
        )
        environment_state = self.environment.step(
            state.environment,
            joint_targets,
        )
        phases = jp.fmod(state.phases + self.phase_delta + jp.pi, 2.0 * jp.pi) - jp.pi
        return state.replace(
            environment=environment_state,
            commands=strategy.command,
            previous_actions=locomotion.action,
            phases=phases,
            strategy_values=strategy.value,
            locomotion_values=locomotion.value,
            strategy_log_probabilities=strategy_log_probability,
            strategy_entropies=strategy_entropy,
            locomotion_log_probabilities=locomotion_log_probability,
            locomotion_entropies=locomotion_entropy,
            policy_rng=next_policy_key,
            locomotion_steps=state.locomotion_steps + 1,
            strategy_updates=(
                state.strategy_updates + should_update_strategy.astype(jp.int32)
            ),
        )
