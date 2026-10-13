"""Create publication-quality tables and plots from hierarchical TAG evaluations."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/motionforge-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, PercentFormatter

from motionforge.cli import run_hydra


@dataclass
class Config:
    evaluation_directory: Path = Path("logs/hierarchical/evaluation")
    pattern: str = "*.json"
    output_directory: Path = Path("logs/hierarchical/plots")
    trajectory_pattern: str = "*baseline*.json"
    dpi: int = 180


def _scenario(report: dict, role: str) -> dict:
    return next(item for item in report["scenarios"] if item["role"] == role)


def _condition_label(report: dict) -> str:
    condition = str(report["_condition"])
    action_mode = "Stochastic" if condition.endswith("_stochastic") else "Deterministic"
    condition = condition.removesuffix("_stochastic").removesuffix("_deterministic")
    pairings = {
        "historical_vs_baseline": "iter 10 → iter 20",
        "baseline_vs_historical": "iter 20 → iter 10",
        "baseline_vs_warm_start": "iter 20 → warm start",
        "later_seed0_vs_baseline": "iter 145 seed 0 → iter 20",
        "later_seed1_vs_baseline": "iter 145 seed 1 → iter 20",
        "later_seed2_vs_baseline": "iter 145 seed 2 → iter 20",
        "later_seed3_vs_baseline": "iter 145 seed 3 → iter 20",
        "baseline_vs_later_seed0": "iter 20 → iter 145 seed 0",
        "baseline_vs_later_seed1": "iter 20 → iter 145 seed 1",
        "baseline_vs_later_seed2": "iter 20 → iter 145 seed 2",
        "baseline_vs_later_seed3": "iter 20 → iter 145 seed 3",
    }
    pairing = pairings.get(condition, condition.replace("_", " "))
    return f"{pairing}\n{action_mode}"


def _rows(reports: list[dict]) -> list[dict[str, object]]:
    rows = []
    for report in reports:
        for role in ("pursuer", "evader"):
            scenario = _scenario(report, role)
            rows.append(
                {
                    "condition": report["_condition"],
                    "checkpoint": report["checkpoint"],
                    "iteration": report["checkpoint_iteration"],
                    "action_mode": report.get("action_mode", "deterministic"),
                    "opponent_checkpoint": report.get("opponent", {}).get("checkpoint"),
                    "opponent_category": report.get("opponent", {}).get(
                        "category", "unknown"
                    ),
                    "role": role,
                    "role_success_rate": scenario["role_success_rate"],
                    "physical_survival_rate": scenario["physical_survival_rate"],
                    "mean_command_tracking_rmse": scenario[
                        "mean_command_tracking_rmse"
                    ],
                    "mean_final_distance": scenario["mean_final_distance"],
                }
            )
    return rows


def _save_figure(fig: plt.Figure, destination: Path, dpi: int) -> None:
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _comparison_figure(reports: list[dict]) -> plt.Figure:
    ordered = sorted(
        reports,
        key=lambda report: (
            report["checkpoint_iteration"],
            report.get("action_mode", "deterministic"),
            report["_condition"],
        ),
    )
    labels = [_condition_label(report) for report in ordered]
    x = np.arange(len(ordered))
    width = 0.38
    colors = {"pursuer": "#2563eb", "evader": "#dc2626"}
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=True)
    panels = (
        ("role_success_rate", "Role success rate", (0.0, 1.05)),
        ("physical_survival_rate", "Physical survival rate", (0.0, 1.05)),
        ("mean_command_tracking_rmse", "Command tracking RMSE", (0.0, 0.5)),
    )
    for axis, (field, title, limits) in zip(axes, panels, strict=True):
        for role, offset in (("pursuer", -width / 2), ("evader", width / 2)):
            values = [float(_scenario(report, role)[field]) for report in ordered]
            axis.bar(
                x + offset,
                values,
                width,
                label=role.capitalize(),
                color=colors[role],
                alpha=0.86,
                edgecolor="white",
                linewidth=0.5,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel("RMSE" if "RMSE" in title else "Rate")
        axis.set_ylim(*limits)
        axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if "rate" in title.lower():
            axis.yaxis.set_major_locator(MultipleLocator(0.25))
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        else:
            axis.yaxis.set_major_locator(MultipleLocator(0.1))
    axes[0].legend(loc="lower right", ncols=2, frameon=False)
    axes[-1].set_xticks(x, labels, rotation=42, ha="right", fontsize=8)
    axes[-1].set_xlabel("Evaluation condition")
    fig.suptitle(
        "Hierarchical TAG held-out cross-play (16 seeds per role)",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def _trajectory_figure(report: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for column, role in enumerate(("pursuer", "evader")):
        scenario = _scenario(report, role)
        distance_axis, path_axis = axes[0, column], axes[1, column]
        for rollout in scenario["rollouts"]:
            trajectory = rollout.get("trajectory")
            if trajectory:
                distance_axis.plot(
                    trajectory["time_seconds"],
                    trajectory["distance"],
                    linewidth=1.3,
                    alpha=0.65,
                )
        distance_axis.axhline(
            0.4,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            label="tag threshold",
        )
        distance_axis.set_title(
            f"{role.capitalize()}: opponent distance", loc="left", fontweight="bold"
        )
        distance_axis.set_xlabel("Time (s)")
        distance_axis.set_ylabel("Distance (m)")
        distance_axis.set_xlim(left=0.0)
        distance_axis.set_ylim(bottom=0.0)
        distance_axis.xaxis.set_major_locator(MultipleLocator(1.0))
        distance_axis.yaxis.set_major_locator(MultipleLocator(1.0))
        distance_axis.grid(color="#d1d5db", linewidth=0.8, alpha=0.8)
        distance_axis.spines[["top", "right"]].set_visible(False)
        distance_axis.legend(loc="best", frameon=False)

        first = scenario["rollouts"][0].get("trajectory")
        if first:
            learner = np.asarray(first["learner_position"])
            opponent = np.asarray(first["opponent_position"])
            for points, color, label in (
                (learner, "#2563eb", "learner"),
                (opponent, "#dc2626", "opponent"),
            ):
                path_axis.plot(
                    points[:, 0], points[:, 1], color=color, linewidth=2.0, label=label
                )
                path_axis.scatter(
                    points[0, 0], points[0, 1], color=color, marker="o", s=35, zorder=3
                )
                path_axis.scatter(
                    points[-1, 0],
                    points[-1, 1],
                    color=color,
                    marker="x",
                    s=45,
                    zorder=3,
                )
        path_axis.set_title(
            f"{role.capitalize()}: seed {scenario['rollouts'][0]['seed']} planar path",
            loc="left",
            fontweight="bold",
        )
        path_axis.set_xlabel("World x (m)")
        path_axis.set_ylabel("World y (m)")
        path_axis.set_xlim(-5.0, 5.0)
        path_axis.set_ylim(-5.0, 5.0)
        path_axis.xaxis.set_major_locator(MultipleLocator(1.0))
        path_axis.yaxis.set_major_locator(MultipleLocator(1.0))
        path_axis.set_aspect("equal", adjustable="box")
        path_axis.grid(color="#d1d5db", linewidth=0.8, alpha=0.8)
        path_axis.spines[["top", "right"]].set_visible(False)
        path_axis.legend(loc="best", frameon=False)
    fig.suptitle(
        f"{_condition_label(report)} — iteration {report['checkpoint_iteration']}",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def _load_reports(config: Config) -> tuple[list[dict], dict[Path, dict]]:
    reports, by_path = [], {}
    for path in sorted(config.evaluation_directory.glob(config.pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") == "hierarchical_tag_checkpoint_evaluation":
            payload["_condition"] = path.stem
            reports.append(payload)
            by_path[path] = payload
    return reports, by_path


def main(config: Config) -> None:
    reports, by_path = _load_reports(config)
    if not reports:
        raise FileNotFoundError("no hierarchical TAG evaluation reports matched")
    config.output_directory.mkdir(parents=True, exist_ok=True)
    rows = _rows(reports)
    with (config.output_directory / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _save_figure(
        _comparison_figure(reports),
        config.output_directory / "crossplay_summary",
        config.dpi,
    )
    trajectory_count = 0
    for path in sorted(config.evaluation_directory.glob(config.trajectory_pattern)):
        report = by_path.get(path)
        if report is not None:
            _save_figure(
                _trajectory_figure(report),
                config.output_directory / f"{path.stem}_trajectories",
                config.dpi,
            )
            trajectory_count += 1
    result = {
        "evaluation_count": len(reports),
        "output_directory": str(config.output_directory),
        "summary_csv": str(config.output_directory / "summary.csv"),
        "summary_png": str(config.output_directory / "crossplay_summary.png"),
        "summary_svg": str(config.output_directory / "crossplay_summary.svg"),
        "trajectory_plot_count": trajectory_count,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
