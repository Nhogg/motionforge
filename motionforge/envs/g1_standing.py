"""g1_standing.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    G1 joystick environment with structured locomotion command sampling.

    MotionForge retrains MujoCo Playground's G1 physics observations, rewards,
    and actions while replacing its continuous mixed-command sampler.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src.locomotion.g1 import joystick

COMMAND_OBSERVATION_SLICE = slice(9, 12)


def default_config() -> config_dict.ConfigDict:
    """Return MotionForge's structured-command G1 configuration."""
    config = joystick.default_config()

    config.standing_probability = 0.30
    config.pure_x_probability = 0.15
    config.pure_y_probability = 0.10
    config.pure_yaw_probability = 0.20
    config.mixed_probability = 0.25

    config.rapid_command_transitions = False
    config.rapid_command_episode_probability = 0.25
    config.command_transition_interval_min = 1.0
    config.command_transition_interval_max = 3.0
    config.command_reversal_probability = 0.50

    return config


class G1StandingJoystick(joystick.Joystick):
    """Flat-terrain G1 environment with structured velocity commands."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        task: str = "flat_terrain",
    ) -> None:
        super().__init__(
            task=task,
            config=default_config() if config is None else config,
            config_overrides=config_overrides,
        )

        probabilities = (
            float(self._config.standing_probability),
            float(self._config.pure_x_probability),
            float(self._config.pure_y_probability),
            float(self._config.pure_yaw_probability),
            float(self._config.mixed_probability),
        )

        if any(probability < 0.0 for probability in probabilities):
            raise ValueError("Command-mode probabilities must be non-negative")

        if abs(sum(probabilities) - 1.0) > 1e-9:
            raise ValueError(
                f"Command-mode probabilities must sum to one; got {sum(probabilities)}"
            )

        self._command_mode_probabilities = jp.asarray(probabilities)

        if self._config.command_transition_interval_min <= 0.0:
            raise ValueError("command_transition_interval_min must be positive")
        if (
            self._config.command_transition_interval_max
            < self._config.command_transition_interval_min
        ):
            raise ValueError(
                "command_transition_interval_max must be greater than or equal "
                "to command_transition_interval_min"
            )
        if not 0.0 <= self._config.command_reversal_probability <= 1.0:
            raise ValueError("command_reversal_probability must be in [0, 1]")
        if not 0.0 <= self._config.rapid_command_episode_probability <= 1.0:
            raise ValueError("rapid_command_episode_probability must be in [0, 1]")

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample standing, axial, pure-yaw, or mixed commands."""
        mode_rng, x_rng, y_rng, yaw_rng = jax.random.split(rng, 4)

        x = jax.random.uniform(
            x_rng,
            minval=self._config.lin_vel_x[0],
            maxval=self._config.lin_vel_x[1],
        )
        y = jax.random.uniform(
            y_rng,
            minval=self._config.lin_vel_y[0],
            maxval=self._config.lin_vel_y[1],
        )
        yaw = jax.random.uniform(
            yaw_rng,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        zero = jp.zeros((), dtype=x.dtype)

        candidates = jp.stack(
            [
                jp.array([zero, zero, zero]),
                jp.array([x, zero, zero]),
                jp.array([zero, y, zero]),
                jp.array([zero, zero, yaw]),
                jp.array([x, y, yaw]),
            ]
        )

        mode = jax.random.choice(
            mode_rng,
            candidates.shape[0],
            p=self._command_mode_probabilities,
        )
        return candidates[mode]

    def _sample_transition_interval(self, rng: jax.Array) -> jax.Array:
        """Sample a command hold duration expressed in control steps."""
        duration = jax.random.uniform(
            rng,
            minval=self._config.command_transition_interval_min,
            maxval=self._config.command_transition_interval_max,
        )
        return jp.maximum(jp.round(duration / self.dt).astype(jp.int32), 1)

    def reset(self, rng: jax.Array):
        """Initialize optional rapid-command curriculum bookkeeping."""
        state = super().reset(rng)
        if not self._config.rapid_command_transitions:
            return state

        info = dict(state.info)
        info["rng"], active_rng, interval_rng = jax.random.split(info["rng"], 3)
        info["rapid_command_episode"] = (
            jax.random.uniform(active_rng)
            < self._config.rapid_command_episode_probability
        )
        info["command_transition_step"] = jp.zeros((), dtype=jp.int32)
        info["command_transition_interval"] = self._sample_transition_interval(
            interval_rng
        )
        return state.replace(info=info)

    def step(self, state, action: jax.Array):
        """Advance physics and optionally resample or reverse commands."""
        state = super().step(state, action)
        if not self._config.rapid_command_transitions:
            return state

        info = dict(state.info)
        transition_step = info["command_transition_step"] + 1
        should_transition = info["rapid_command_episode"] & (
            transition_step >= info["command_transition_interval"]
        )

        info["rng"], sample_rng, reverse_rng, interval_rng = jax.random.split(
            info["rng"],
            4,
        )
        sampled_command = self.sample_command(sample_rng)
        reversed_command = -info["command"]
        reversal_available = jp.linalg.norm(info["command"]) > 1e-6
        choose_reversal = reversal_available & (
            jax.random.uniform(reverse_rng) < self._config.command_reversal_probability
        )
        transition_command = jp.where(
            choose_reversal,
            reversed_command,
            sampled_command,
        )
        command = jp.where(
            should_transition,
            transition_command,
            info["command"],
        )

        info["command"] = command
        info["command_transition_step"] = jp.where(
            should_transition,
            0,
            transition_step,
        )
        info["command_transition_interval"] = jp.where(
            should_transition,
            self._sample_transition_interval(interval_rng),
            info["command_transition_interval"],
        )
        info["step"] = jp.where(should_transition, 0, info["step"])

        observation = dict(state.obs)
        for name in ("state", "privileged_state"):
            observation[name] = (
                observation[name].at[COMMAND_OBSERVATION_SLICE].set(command)
            )

        return state.replace(info=info, obs=observation)
