"""Reproducible high-level command schedules for non-TAG baselines."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp

AGGRESSIVE_MANEUVER_NAMES = (
    "hard_turn",
    "forward_reverse",
    "braking",
    "lateral_reverse",
    "high_yaw_reverse",
)


@dataclass(frozen=True)
class AggressiveCommandSchedule:
    """Cyclic paired-command schedule with guaranteed maneuver coverage."""

    hold_steps: int
    seed: int

    def __post_init__(self) -> None:
        if self.hold_steps <= 0:
            raise ValueError("hold_steps must be positive")

    @property
    def cycle_steps(self) -> int:
        return len(AGGRESSIVE_MANEUVER_NAMES) * 2 * self.hold_steps

    def command(self, sample_index: int) -> tuple[jax.Array, str, int]:
        """Return command, maneuver name, and phase for a global sample index."""
        if sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        segment = (sample_index // self.hold_steps) % (
            len(AGGRESSIVE_MANEUVER_NAMES) * 2
        )
        maneuver_offset = self.seed % len(AGGRESSIVE_MANEUVER_NAMES)
        maneuver_index = (segment // 2 + maneuver_offset) % len(
            AGGRESSIVE_MANEUVER_NAMES
        )
        phase = segment % 2
        direction = -1.0 if ((self.seed // len(AGGRESSIVE_MANEUVER_NAMES)) % 2) else 1.0

        pairs = jp.asarray(
            [
                [[0.6, 0.0, direction], [0.6, 0.0, -direction]],
                [[0.8 * direction, 0.0, 0.0], [-0.8 * direction, 0.0, 0.0]],
                [[0.8 * direction, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.5 * direction, 0.0], [0.0, -0.5 * direction, 0.0]],
                [[0.0, 0.0, direction], [0.0, 0.0, -direction]],
            ],
            dtype=jp.float32,
        )
        return (
            pairs[maneuver_index, phase],
            AGGRESSIVE_MANEUVER_NAMES[maneuver_index],
            phase,
        )
