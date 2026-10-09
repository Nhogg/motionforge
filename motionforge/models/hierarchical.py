"""Role-conditioned hierarchical policy for two-agent locomotion tasks.

The strategy and locomotion modules are sequentially conditioned but retain
separate parameter subtrees.  The command passed to locomotion is explicitly
stop-gradient so a physical locomotion objective cannot update strategy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jp
from flax import linen as nn
from flax import struct

STRATEGY_OBSERVATION_SIZE = 11
STRATEGY_ACTION_SIZE = 3
LOCOMOTION_OBSERVATION_SIZE = 103
LOCOMOTION_ACTION_SIZE = 29


@struct.dataclass
class StrategyObservation:
    """Task-level state visible to the role-conditioned strategy module."""

    opponent_position_local: jax.Array
    opponent_velocity_local: jax.Array
    arena_center_local: jax.Array
    previous_command: jax.Array
    role: jax.Array

    def as_array(self) -> jax.Array:
        """Flatten fields along the final axis while preserving batch axes."""
        return jp.concatenate(
            (
                self.opponent_position_local,
                self.opponent_velocity_local,
                self.arena_center_local,
                self.previous_command,
                self.role,
            ),
            axis=-1,
        )


@struct.dataclass
class LocomotionObservation:
    """Motor-level state with no opponent or game-role information."""

    root_linear_velocity: jax.Array
    root_angular_velocity: jax.Array
    projected_gravity: jax.Array
    joint_position: jax.Array
    joint_velocity: jax.Array
    previous_action: jax.Array
    phase_cosine: jax.Array
    phase_sine: jax.Array

    def as_array(self, command: jax.Array) -> jax.Array:
        """Build the accepted P2 actor observation in its original order."""
        return jp.concatenate(
            (
                self.root_linear_velocity,
                self.root_angular_velocity,
                self.projected_gravity,
                command,
                self.joint_position,
                self.joint_velocity,
                self.previous_action,
                self.phase_cosine,
                self.phase_sine,
            ),
            axis=-1,
        )


@struct.dataclass
class LocomotionNormalization:
    """Non-trainable observation statistics restored from locomotion PPO."""

    mean: jax.Array
    std: jax.Array

    def normalize(self, observation: jax.Array) -> jax.Array:
        return (observation - self.mean) / self.std


@dataclass(frozen=True)
class HierarchicalPolicyConfig:
    """Static neural-network configuration for the first hierarchy version."""

    strategy_hidden_layer_sizes: tuple[int, ...] = (128, 128)
    locomotion_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    command_scale: tuple[float, float, float] = (1.0, 0.5, 1.0)
    strategy_initial_log_std: float = -1.0
    locomotion_initial_log_std: float = -2.0

    def __post_init__(self) -> None:
        if not self.strategy_hidden_layer_sizes:
            raise ValueError("strategy_hidden_layer_sizes must not be empty")
        if not self.locomotion_hidden_layer_sizes:
            raise ValueError("locomotion_hidden_layer_sizes must not be empty")
        if any(size <= 0 for size in self.strategy_hidden_layer_sizes):
            raise ValueError("strategy hidden-layer sizes must be positive")
        if any(size <= 0 for size in self.locomotion_hidden_layer_sizes):
            raise ValueError("locomotion hidden-layer sizes must be positive")
        if len(self.command_scale) != STRATEGY_ACTION_SIZE:
            raise ValueError("command_scale must contain three values")
        if any(scale <= 0.0 for scale in self.command_scale):
            raise ValueError("command scales must be positive")


@struct.dataclass
class StrategyOutput:
    location: jax.Array
    command: jax.Array
    log_std: jax.Array
    value: jax.Array


@struct.dataclass
class LocomotionOutput:
    location: jax.Array
    action: jax.Array
    log_std: jax.Array
    value: jax.Array


@struct.dataclass
class HierarchicalPolicyOutput:
    strategy: StrategyOutput
    locomotion: LocomotionOutput


class StrategyEncoder(nn.Module):
    hidden_layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        features = observation
        for size in self.hidden_layer_sizes:
            features = nn.tanh(nn.Dense(size)(features))
        return features


class StrategyHead(nn.Module):
    command_scale: tuple[float, float, float]
    initial_log_std: float

    @nn.compact
    def __call__(self, features: jax.Array) -> StrategyOutput:
        location = nn.Dense(STRATEGY_ACTION_SIZE, name="command")(features)
        command = nn.tanh(location)
        command = command * jp.asarray(self.command_scale, dtype=command.dtype)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(self.initial_log_std),
            (STRATEGY_ACTION_SIZE,),
        )
        log_std = jp.broadcast_to(log_std, location.shape)
        value = nn.Dense(1, name="value")(features)[..., 0]
        return StrategyOutput(
            location=location,
            command=command,
            log_std=log_std,
            value=value,
        )


class LocomotionEncoder(nn.Module):
    hidden_layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        features = observation
        for size in self.hidden_layer_sizes:
            features = jax.nn.silu(nn.Dense(size)(features))
        return features


class MotorHead(nn.Module):
    initial_log_std: float

    @nn.compact
    def __call__(self, features: jax.Array) -> LocomotionOutput:
        location = nn.Dense(LOCOMOTION_ACTION_SIZE, name="action")(features)
        action = nn.tanh(location)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(self.initial_log_std),
            (LOCOMOTION_ACTION_SIZE,),
        )
        log_std = jp.broadcast_to(log_std, location.shape)
        value = nn.Dense(1, name="value")(features)[..., 0]
        return LocomotionOutput(
            location=location,
            action=action,
            log_std=log_std,
            value=value,
        )


class StrategyModule(nn.Module):
    hidden_layer_sizes: Sequence[int]
    command_scale: tuple[float, float, float]
    initial_log_std: float = -1.0

    @nn.compact
    def __call__(self, observation: jax.Array) -> StrategyOutput:
        features = StrategyEncoder(
            self.hidden_layer_sizes,
            name="encoder",
        )(observation)
        return StrategyHead(
            self.command_scale,
            self.initial_log_std,
            name="head",
        )(features)


class LocomotionModule(nn.Module):
    hidden_layer_sizes: Sequence[int]
    initial_log_std: float = -2.0

    @nn.compact
    def __call__(self, observation: jax.Array) -> LocomotionOutput:
        features = LocomotionEncoder(
            self.hidden_layer_sizes,
            name="encoder",
        )(observation)
        return MotorHead(
            self.initial_log_std,
            name="motor_head",
        )(features)


class HierarchicalPolicy(nn.Module):
    """Sequential strategy-to-locomotion policy with isolated gradients."""

    config: HierarchicalPolicyConfig = HierarchicalPolicyConfig()

    @nn.compact
    def __call__(
        self,
        strategy_observation: StrategyObservation,
        locomotion_observation: LocomotionObservation,
        locomotion_normalization: LocomotionNormalization | None = None,
    ) -> HierarchicalPolicyOutput:
        strategy_array = strategy_observation.as_array()
        strategy = StrategyModule(
            self.config.strategy_hidden_layer_sizes,
            self.config.command_scale,
            self.config.strategy_initial_log_std,
            name="strategy",
        )(strategy_array)

        motor_command = jax.lax.stop_gradient(strategy.command)
        locomotion_array = locomotion_observation.as_array(motor_command)
        if locomotion_normalization is not None:
            locomotion_array = locomotion_normalization.normalize(locomotion_array)
        locomotion = LocomotionModule(
            self.config.locomotion_hidden_layer_sizes,
            self.config.locomotion_initial_log_std,
            name="locomotion",
        )(locomotion_array)
        return HierarchicalPolicyOutput(
            strategy=strategy,
            locomotion=locomotion,
        )
