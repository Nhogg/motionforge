"""Validate role-conservative hierarchical checkpoint ranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.evaluation import select_hierarchical_checkpoint


@dataclass
class Config:
    output: Path = Path("logs/hierarchical/checkpoint_selection_a.json")


def _evaluation(
    iteration: int,
    pursuer_success: float,
    evader_success: float,
    pursuer_survival: float,
    evader_survival: float,
    tracking: float,
) -> dict[str, object]:
    return {
        "checkpoint": f"/checkpoints/{iteration:012d}",
        "checkpoint_iteration": iteration,
        "experiment": "hierarchical_tag_checkpoint_evaluation",
        "passed": True,
        "scenarios": [
            {
                "mean_command_tracking_rmse": tracking,
                "physical_survival_rate": pursuer_survival,
                "role": "pursuer",
                "role_success_rate": pursuer_success,
            },
            {
                "mean_command_tracking_rmse": tracking,
                "physical_survival_rate": evader_survival,
                "role": "evader",
                "role_success_rate": evader_success,
            },
        ],
    }


def main(config: Config) -> None:
    evaluations = [
        _evaluation(10, 1.0, 1.0, 1.0, 1.0, 0.12),
        _evaluation(20, 1.0, 0.5, 1.0, 1.0, 0.10),
        _evaluation(30, 0.75, 1.0, 1.0, 1.0, 0.09),
        _evaluation(40, 1.0, 1.0, 0.75, 1.0, 0.08),
    ]
    selected, ranked = select_hierarchical_checkpoint(evaluations)
    checks = {
        "minimum_role_success_has_priority": selected.iteration == 10,
        "weak_role_not_hidden_by_mean": ranked[-1].iteration == 20,
        "survival_breaks_success_tie": ranked[1].iteration == 40,
        "all_candidates_retained": len(ranked) == len(evaluations),
    }
    result = {
        "checks": checks,
        "experiment": "hierarchical_checkpoint_selection_test",
        "passed": all(checks.values()),
        "ranking": [candidate.iteration for candidate in ranked],
        "selected_iteration": selected.iteration,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
