"""Role-separated TAG return and fixed-reference-calibrated diagnostic plots.

Missing measurements remain missing; success-only legacy Elo is never substituted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/motionforge-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

from motionforge.logging.wandb import WandbMetricsLogger


def progress_series(history: list[dict], role: str) -> dict[str, list[float]]:
    if role not in ("pursuer", "evader"):
        raise ValueError("Benchmark series must remain role separated")
    keys = {
        "steps": "environment_steps",
        "reward": f"strategy/{role}/episode_return_mean",
        "skill": f"benchmark/{role}/relative_skill",
        "difficulty": f"benchmark/{role}/opponent_difficulty",
        "completion": f"benchmark/{role}/observed_completion",
    }
    return {
        name: [float(row.get(key, float("nan"))) for row in history]
        for name, key in keys.items()
    }


@dataclass
class TagProgressPlotCallback:
    logger: WandbMetricsLogger
    output_dir: Path
    history: list[dict[str, float]] = field(default_factory=list)

    def __call__(self, environment_steps: int, metrics: dict[str, float]) -> None:
        self.history.append({**metrics, "environment_steps": float(environment_steps)})
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for role in ("pursuer", "evader", "combined"):
            path = plot_dir / f"{role}_training_progress.png"
            self._plot_role(role, path)
            self.logger.log_image(
                environment_steps=environment_steps,
                path=path,
                caption=(
                    f"{role}: shaped return; online relative skill calibrated to a "
                    "frozen reference; difficulty is that reference's failure rate "
                    "against the sampled opponent, not 1 minus learner skill. "
                    "Unknown challenges remain missing. Not a game-win probability."
                ),
                key=f"progress/{role}",
            )

    def _plot_role(self, role: str, path: Path) -> None:
        roles = ("pursuer", "evader") if role == "combined" else (role,)
        figure, axes = plt.subplots(
            len(roles), 1, figsize=(12.5, 5.8 * len(roles)), squeeze=False
        )
        for current_role, reward_axis in zip(roles, axes[:, 0], strict=True):
            values = progress_series(self.history, current_role)
            probability_axis = reward_axis.twinx()
            reward_axis.plot(
                values["steps"],
                values["reward"],
                color="#2563eb",
                linewidth=2,
                label="Shaped episode return",
            )
            for name, color, label in (
                ("skill", "#16a34a", "Relative skill estimate (fixed reference = 50%)"),
                ("difficulty", "#dc2626", "Difficulty (fixed-reference failure rate)"),
                ("completion", "#6b7280", "Observed window completion"),
            ):
                if np.any(np.isfinite(values[name])):
                    probability_axis.plot(
                        values["steps"],
                        values[name],
                        color=color,
                        linewidth=1 if name == "completion" else 2,
                        alpha=0.5 if name == "completion" else 1,
                        linestyle=":" if name == "completion" else "-",
                        label=label,
                    )
            if not np.any(np.isfinite(values["skill"])):
                reward_axis.text(
                    0.02,
                    0.94,
                    "Skill / difficulty not calibrated — replay required",
                    transform=reward_axis.transAxes,
                    va="top",
                    color="#b91c1c",
                )
            reward_axis.set_title(
                f"Hierarchical TAG — {current_role.title()}", loc="left"
            )
            reward_axis.set_xlabel("Environment steps")
            reward_axis.set_ylabel("Shaped strategy return (not terminal payoff)")
            probability_axis.set_ylabel(
                "Reference-relative estimate / completion fraction"
            )
            reward_axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value / 1e6:.1f}M")
            )
            probability_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            probability_axis.set_ylim(0, 1)
            reward_axis.grid(alpha=0.25)
            left, left_labels = reward_axis.get_legend_handles_labels()
            right, right_labels = probability_axis.get_legend_handles_labels()
            reward_axis.legend(left + right, left_labels + right_labels, loc="best")
        figure.suptitle(
            "Finite-window benchmarks — role separated; no cancelling combined rating",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(figure)
