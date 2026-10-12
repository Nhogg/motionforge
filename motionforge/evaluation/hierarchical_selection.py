"""Role-conservative selection for hierarchical TAG checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HierarchicalCheckpointCandidate:
    checkpoint: str
    iteration: int
    minimum_role_success: float
    minimum_physical_survival: float
    mean_role_success: float
    mean_physical_survival: float
    mean_pursuer_tag_time_seconds: float
    maximum_tracking_rmse: float

    @property
    def rank(self) -> tuple[float, ...]:
        """Return a lexicographic rank in descending preference order."""
        return (
            self.minimum_role_success,
            self.minimum_physical_survival,
            self.mean_role_success,
            self.mean_physical_survival,
            -self.mean_pursuer_tag_time_seconds,
            -self.maximum_tracking_rmse,
            -float(self.iteration),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "iteration": self.iteration,
            "maximum_tracking_rmse": self.maximum_tracking_rmse,
            "mean_physical_survival": self.mean_physical_survival,
            "mean_pursuer_tag_time_seconds": self.mean_pursuer_tag_time_seconds,
            "mean_role_success": self.mean_role_success,
            "minimum_physical_survival": self.minimum_physical_survival,
            "minimum_role_success": self.minimum_role_success,
            "rank": list(self.rank),
        }


def hierarchical_checkpoint_candidate(
    evaluation: Mapping[str, Any],
) -> HierarchicalCheckpointCandidate:
    """Validate one evaluator payload and derive its conservative metrics."""
    if evaluation.get("experiment") != "hierarchical_tag_checkpoint_evaluation":
        raise ValueError("evaluation has the wrong experiment type")
    if not evaluation.get("passed", False):
        raise ValueError("evaluation did not pass its integrity checks")
    scenarios = evaluation.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 2:
        raise ValueError("evaluation must contain exactly two role scenarios")
    by_role = {scenario.get("role"): scenario for scenario in scenarios}
    if set(by_role) != {"pursuer", "evader"}:
        raise ValueError("evaluation must contain pursuer and evader scenarios")

    def values(field: str) -> list[float]:
        result = [float(by_role[role][field]) for role in ("pursuer", "evader")]
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"scenario field {field} must be finite")
        return result

    success = values("role_success_rate")
    survival = values("physical_survival_rate")
    tracking = values("mean_command_tracking_rmse")
    for name, items in (("success", success), ("survival", survival)):
        if not all(0.0 <= value <= 1.0 for value in items):
            raise ValueError(f"{name} rates must be in [0, 1]")
    if not all(value >= 0.0 for value in tracking):
        raise ValueError("tracking RMSE must be nonnegative")

    control_timestep = float(evaluation["control_timestep"])
    rollout_steps = int(evaluation["rollout_steps"])
    if not math.isfinite(control_timestep) or control_timestep <= 0.0:
        raise ValueError("control_timestep must be finite and positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    pursuer_rollouts = by_role["pursuer"].get("rollouts")
    if not isinstance(pursuer_rollouts, list) or not pursuer_rollouts:
        raise ValueError("pursuer scenario must contain nonempty rollouts")
    pursuer_tag_steps = [
        min(int(rollout["steps_alive"]), rollout_steps)
        if bool(rollout["role_success"])
        else rollout_steps
        for rollout in pursuer_rollouts
    ]
    if not all(0 < steps <= rollout_steps for steps in pursuer_tag_steps):
        raise ValueError("pursuer rollout steps_alive must be in the rollout horizon")
    mean_pursuer_tag_time_seconds = (
        sum(pursuer_tag_steps) / len(pursuer_tag_steps) * control_timestep
    )

    return HierarchicalCheckpointCandidate(
        checkpoint=str(evaluation["checkpoint"]),
        iteration=int(evaluation["checkpoint_iteration"]),
        minimum_role_success=min(success),
        minimum_physical_survival=min(survival),
        mean_role_success=sum(success) / 2.0,
        mean_physical_survival=sum(survival) / 2.0,
        mean_pursuer_tag_time_seconds=mean_pursuer_tag_time_seconds,
        maximum_tracking_rmse=max(tracking),
    )


def select_hierarchical_checkpoint(
    evaluations: Sequence[Mapping[str, Any]],
) -> tuple[HierarchicalCheckpointCandidate, list[HierarchicalCheckpointCandidate]]:
    """Rank valid checkpoint evaluations without averaging away a weak role."""
    if not evaluations:
        raise ValueError("at least one evaluation is required")
    candidates = [hierarchical_checkpoint_candidate(item) for item in evaluations]
    checkpoints = [candidate.checkpoint for candidate in candidates]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoint evaluations must be unique")
    ranked = sorted(candidates, key=lambda candidate: candidate.rank, reverse=True)
    return ranked[0], ranked
