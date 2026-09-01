from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src.locomotion.g1 import joystick


def default_config() -> config_dict.ConfigDict:
    """Return the G1 configuration owned by MotionForge."""
    config = joystick.default_config()
    config.standing_probability = 0.30
    return config


class G1StandingJoystick(joystick.Joystick):
    """Flat-terrain G1 environment with explicit standing commands."""

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

    def sample_command(self, rng: jax.Array) -> jax.Array:
        standing_rng, x_rng, y_rng, yaw_rng = jax.random.split(rng, 4)

        moving_command = jp.array(
            [
                jax.random.uniform(
                    x_rng,
                    minval=self._config.lin_vel_x[0],
                    maxval=self._config.lin_vel_x[1],
                ),
                jax.random.uniform(
                    y_rng,
                    minval=self._config.lin_vel_y[0],
                    maxval=self._config.lin_vel_y[1],
                ),
                jax.random.uniform(
                    yaw_rng,
                    minval=self._config.ang_vel_yaw[0],
                    maxval=self._config.ang_vel_yaw[1],
                ),
            ]
        )

        standing = jax.random.bernoulli(
            standing_rng,
            p=self._config.standing_probability,
        )

        return jp.where(
            standing,
            jp.zeros_like(moving_command),
            moving_command,
        )
