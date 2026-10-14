"""Live, publication-readable TAG reward/skill/difficulty plots."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/motionforge-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

from motionforge.logging.wandb import WandbMetricsLogger


@dataclass
class TagProgressPlotCallback:
    logger: WandbMetricsLogger
    output_dir: Path
    history: list[dict[str, float]] = field(default_factory=list)

    def __call__(self, environment_steps: int, metrics: dict[str, float]) -> None:
        row = {"environment_steps": float(environment_steps), **metrics}
        self.history.append(row)
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for role in ("pursuer", "evader", "combined"):
            path = plot_dir / f"{role}_training_progress.png"
            self._plot_role(role, path)
            self.logger.log_image(
                environment_steps=environment_steps,
                path=path,
                caption=(
                    f"{role.title()} reward, Bradley–Terry skill, and matchup "
                    f"difficulty through {environment_steps:,} environment steps"
                ),
                key=f"progress/{role}",
            )

    def _plot_role(self, role: str, path: Path) -> None:
        steps = [row["environment_steps"] for row in self.history]
        if role == "combined":
            rewards = [
                0.5
                * (
                    row["strategy/pursuer/episode_return_mean"]
                    + row["strategy/evader/episode_return_mean"]
                    + 1.0
                )
                for row in self.history
            ]
        else:
            rewards = [
                row[f"strategy/{role}/episode_return_mean"] for row in self.history
            ]
        running_maximum = []
        for reward in rewards:
            running_maximum.append(
                reward if not running_maximum else max(running_maximum[-1], reward)
            )
        skill = [row[f"elo/{role}/perceived_skill"] for row in self.history]
        difficulty = [row[f"elo/{role}/task_difficulty"] for row in self.history]

        figure, reward_axis = plt.subplots(figsize=(12.5, 7.0))
        probability_axis = reward_axis.twinx()
        reward_axis.plot(
            steps,
            rewards,
            color="#2563eb",
            linewidth=2.4,
            marker="o",
            label="Episode reward",
        )
        reward_axis.plot(
            steps,
            running_maximum,
            color="#1e3a8a",
            linestyle="--",
            linewidth=1.8,
            label="Running maximum reward",
        )
        probability_axis.plot(
            steps,
            skill,
            color="#16a34a",
            linewidth=2.3,
            marker="o",
            label="Perceived skill (Elo/BT)",
        )
        probability_axis.plot(
            steps,
            difficulty,
            color="#dc2626",
            linewidth=2.3,
            marker="o",
            label="Matchup difficulty (Elo/BT)",
        )
        reward_axis.set_title(
            f"Hierarchical TAG — {role.title()} training progression",
            loc="left",
            fontsize=16,
            fontweight="bold",
        )
        reward_axis.set_xlabel("Environment steps")
        reward_axis.set_ylabel("Terminal episode reward")
        probability_axis.set_ylabel("Probability / normalized rating")
        reward_axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 1e6:.1f}M")
        )
        probability_axis.yaxis.set_major_locator(MultipleLocator(0.1))
        probability_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        probability_axis.set_ylim(0.0, 1.0)
        reward_axis.grid(color="#d1d5db", linewidth=0.8, alpha=0.85)
        reward_axis.set_axisbelow(True)
        reward_axis.spines["top"].set_visible(False)
        probability_axis.spines["top"].set_visible(False)
        left_handles, left_labels = reward_axis.get_legend_handles_labels()
        right_handles, right_labels = probability_axis.get_legend_handles_labels()
        reward_axis.legend(
            left_handles + right_handles,
            left_labels + right_labels,
            loc="best",
            framealpha=0.95,
        )
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight", dpi=180)
        plt.close(figure)
