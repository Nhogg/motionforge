"""Multirate execution for the role-conditioned hierarchical policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jp
from flax import struct

from motionforge.controllers import build_g1_tag_policy_observation_layout
from motionforge.envs import (
    HierarchicalObservationConfig,
    MultiAgentTaskEnv,
    MultiAgentTaskState,
    locomotion_observations,
    role_encodings,
    strategy_observations,
)
from motionforge.models import (
    HierarchicalPolicy,
    LocomotionModule,
    LocomotionNormalization,
    StrategyModule,
    StrategyOutput,
)


@dataclass(frozen=True)
class HierarchicalRolloutConfig:
    physics_hz: int = 500
    locomotion_hz: int = 50
    strategy_hz: int = 10
    gait_frequency_hz: float = 1.375
    action_scale: float = 0.5

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
    locomotion_steps: jax.Array
    strategy_updates: jax.Array


class HierarchicalRolloutRunner:
    """Execute one shared hierarchy for both agents at fixed temporal rates."""

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
        environment_state = self.environment.reset(key)
        return HierarchicalRolloutState(
            environment=environment_state,
            commands=jp.zeros((2, 3), dtype=jp.float32),
            previous_actions=jp.zeros((2, 29), dtype=jp.float32),
            phases=jp.asarray([[0.0, jp.pi], [0.0, jp.pi]], dtype=jp.float32),
            roles=role_encodings(jp.asarray(self.environment.config.pursuer_index)),
            strategy_values=jp.zeros((2,), dtype=jp.float32),
            locomotion_values=jp.zeros((2,), dtype=jp.float32),
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

    def step(
        self,
        parameters: Any,
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
        )
        should_update_strategy = (
            state.locomotion_steps
            % self.config.locomotion_steps_per_strategy_step
            == 0
        )

        def update_strategy(_):
            return strategy_module.apply(
                {"params": parameters["strategy"]},
                strategy_observation.as_array(),
            )

        def hold_strategy(_):
            return StrategyOutput(
                command=state.commands,
                value=state.strategy_values,
            )

        strategy = jax.lax.cond(
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
        locomotion = LocomotionModule(
            self.policy.config.locomotion_hidden_layer_sizes
        ).apply(
            {"params": parameters["locomotion"]},
            normalized_locomotion,
        )
        joint_targets = (
            self.environment.default_joint_targets
            + locomotion.action * self.config.action_scale
        )
        environment_state = self.environment.step(
            state.environment,
            joint_targets,
        )
        phases = (
            jp.fmod(state.phases + self.phase_delta + jp.pi, 2.0 * jp.pi)
            - jp.pi
        )
        return state.replace(
            environment=environment_state,
            commands=strategy.command,
            previous_actions=locomotion.action,
            phases=phases,
            strategy_values=strategy.value,
            locomotion_values=locomotion.value,
            locomotion_steps=state.locomotion_steps + 1,
            strategy_updates=(
                state.strategy_updates + should_update_strategy.astype(jp.int32)
            ),
        )
