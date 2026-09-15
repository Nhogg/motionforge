"""Episode timeout accounting for the two-G1 tag environment.

The timeout is expressed in control steps derived from an explicit duration and
control timestep. Pure JAX functions expose both current-state observation and
advance-then-observe behavior for direct use in the MJX-Warp environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp


@dataclass(frozen=True)
class EpisodeTimeoutConfig:
    duration: float = 20.0
    control_timestep: float = 0.02

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.control_timestep <= 0.0:
            raise ValueError("control_timestep must be positive")

        exact_steps = self.duration / self.control_timestep
        if not math.isclose(exact_steps, round(exact_steps), abs_tol=1e-9):
            raise ValueError(
                "duration must be an integer multiple of control_timestep"
            )

    @property
    def maximum_steps(self) -> int:
        return round(self.duration / self.control_timestep)


_DEFAULT_EPISODE_TIMEOUT_CONFIG = EpisodeTimeoutConfig()


class EpisodeTimeoutObservation(NamedTuple):
    step_count: jax.Array
    elapsed_seconds: jax.Array
    steps_remaining: jax.Array
    timed_out: jax.Array


def observe_episode_timeout(
    step_count: jax.Array,
    config: EpisodeTimeoutConfig = _DEFAULT_EPISODE_TIMEOUT_CONFIG,
) -> EpisodeTimeoutObservation:
    """Describe timeout state without changing the supplied step count."""
    steps = jp.asarray(step_count, dtype=jp.int32)
    steps_remaining = jp.maximum(config.maximum_steps - steps, 0)

    return EpisodeTimeoutObservation(
        step_count=steps,
        elapsed_seconds=steps.astype(jp.float32) * config.control_timestep,
        steps_remaining=steps_remaining,
        timed_out=steps >= config.maximum_steps,
    )


def advance_episode_timeout(
    step_count: jax.Array,
    config: EpisodeTimeoutConfig = _DEFAULT_EPISODE_TIMEOUT_CONFIG,
) -> EpisodeTimeoutObservation:
    """Increment the episode step and return the resulting timeout state."""
    return observe_episode_timeout(
        jp.asarray(step_count, dtype=jp.int32) + 1,
        config,
    )
