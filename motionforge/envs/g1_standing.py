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


def default_config() -> config_dict.ConfigDict:
    """Return MotionForge's structured-command G1 configuration."""
    config = joystick.default_config()

    config.standing_probability = 0.30
    config.pure_x_probability = 0.15
    config.pure_y_probability = 0.10
    config.pure_yaw_probability = 0.20
    config.mixed_probability = 0.25

    return config


class G1StandingJoystick(joystick.Joystick):
    """Flat-terrain G1 environment with structured velocity commands."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            task="flat_terrain",
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
