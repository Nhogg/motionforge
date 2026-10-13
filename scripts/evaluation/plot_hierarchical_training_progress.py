"""Plot reward, opponent difficulty, and perceived skill across TAG training."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/motionforge-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

from motionforge.cli import run_hydra


@dataclass
class Config:
    training_root: Path = Path("logs/hierarchical/training")
    run_prefix: str = "tag_historical_1m_seed"
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    output_directory: Path = Path("logs/hierarchical/plots/training_progress")
    smoothing: float = 0.15
    dpi: int = 200


def _ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _opponent_difficulty(opponent_id: str) -> float:
    match = re.search(r"/(\d{12})-", opponent_id)
    if match is None:
        raise ValueError(f"opponent ID does not contain an iteration: {opponent_id}")
    return int(match.group(1)) / 20.0


def _load_seed(config: Config, seed: int) -> dict[int, dict[str, float]]:
    directories = (
        config.training_root / f"{config.run_prefix}{seed}",
        config.training_root / f"{config.run_prefix}{seed}_resume30",
    )
    records: dict[int, dict[str, float]] = {}
    for directory in directories:
        metrics_path = directory / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            metrics = payload["metrics"]
            iteration = int(metrics["training/iteration"])
            records[iteration] = {
                "environment_steps": float(payload["environment_steps"]),
                "reward": float(metrics["strategy/reward_mean"]),
                "perceived_skill": 0.5
                * (
                    float(metrics["strategy/pursuer_tag_rate"])
                    + float(metrics["strategy/evader_timeout_rate"])
                ),
                "task_difficulty": _opponent_difficulty(payload["opponent_id"]),
            }
    if not records:
        raise FileNotFoundError(f"no metrics found for seed {seed}")
    return records


def _aggregate(config: Config) -> dict[str, np.ndarray]:
    by_seed = [_load_seed(config, seed) for seed in config.seeds]
    iterations = sorted(set.intersection(*(set(records) for records in by_seed)))
    fields = ("environment_steps", "reward", "perceived_skill", "task_difficulty")
    aggregate: dict[str, np.ndarray] = {
        "iteration": np.asarray(iterations, dtype=float)
    }
    for field in fields:
        values = np.asarray(
            [
                [records[iteration][field] for iteration in iterations]
                for records in by_seed
            ]
        )
        if field == "environment_steps":
            aggregate[field] = values.mean(axis=0)
            continue
        smoothed = np.asarray(
            [_ewma(seed_values, config.smoothing) for seed_values in values]
        )
        aggregate[f"{field}_mean"] = smoothed.mean(axis=0)
        aggregate[f"{field}_std"] = smoothed.std(axis=0)
    aggregate["reward_running_max"] = np.maximum.accumulate(aggregate["reward_mean"])
    return aggregate


def _save_table(data: dict[str, np.ndarray], path: Path) -> None:
    fields = list(data)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(zip(*(data[field] for field in fields), strict=True))


def _plot(data: dict[str, np.ndarray]) -> plt.Figure:
    steps = data["environment_steps"]
    reward, reward_std = data["reward_mean"], data["reward_std"]
    skill, skill_std = data["perceived_skill_mean"], data["perceived_skill_std"]
    difficulty = data["task_difficulty_mean"]
    difficulty_std = data["task_difficulty_std"]
    fig, reward_axis = plt.subplots(figsize=(12.5, 7.0))
    skill_axis = reward_axis.twinx()

    reward_axis.plot(
        steps, reward, color="#2563eb", linewidth=2.5, label="Strategy reward (EWMA)"
    )
    reward_axis.fill_between(
        steps,
        reward - reward_std,
        reward + reward_std,
        color="#2563eb",
        alpha=0.16,
        linewidth=0,
        label="Reward ±1 seed SD",
    )
    reward_axis.plot(
        steps,
        data["reward_running_max"],
        color="#1e3a8a",
        linestyle="--",
        linewidth=1.8,
        label="Running maximum observed reward",
    )
    skill_axis.plot(
        steps, difficulty, color="#dc2626", linewidth=2.2, label="Task difficulty proxy"
    )
    skill_axis.fill_between(
        steps,
        np.clip(difficulty - difficulty_std, 0.0, 1.0),
        np.clip(difficulty + difficulty_std, 0.0, 1.0),
        color="#dc2626",
        alpha=0.12,
        linewidth=0,
    )
    skill_axis.plot(
        steps, skill, color="#16a34a", linewidth=2.2, label="Perceived skill proxy"
    )
    skill_axis.fill_between(
        steps,
        np.clip(skill - skill_std, 0.0, 1.0),
        np.clip(skill + skill_std, 0.0, 1.0),
        color="#16a34a",
        alpha=0.15,
        linewidth=0,
    )
    reward_axis.set_title(
        "Hierarchical TAG training progression across four seeds",
        loc="left",
        fontsize=17,
        fontweight="bold",
    )
    reward_axis.set_xlabel("Environment steps")
    reward_axis.set_ylabel("Mean strategy reward per transition")
    skill_axis.set_ylabel("Normalized task difficulty / perceived skill")
    reward_axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1e6:.1f}M")
    )
    reward_axis.xaxis.set_major_locator(MultipleLocator(200_000))
    skill_axis.yaxis.set_major_locator(MultipleLocator(0.2))
    skill_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    skill_axis.set_ylim(0.0, 1.05)
    reward_axis.grid(color="#d1d5db", linewidth=0.8, alpha=0.85)
    reward_axis.set_axisbelow(True)
    reward_axis.spines["top"].set_visible(False)
    skill_axis.spines["top"].set_visible(False)
    handles_left, labels_left = reward_axis.get_legend_handles_labels()
    handles_right, labels_right = skill_axis.get_legend_handles_labels()
    reward_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        ncols=2,
    )
    fig.text(
        0.01,
        0.01,
        "Perceived skill = EWMA of mean(pursuer tag rate, evader timeout rate). "
        "Task difficulty = sampled opponent checkpoint iteration / 20. Shading is ±1 SD across seeds.",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    return fig


def main(config: Config) -> None:
    if not 0.0 < config.smoothing <= 1.0:
        raise ValueError("smoothing must be in (0, 1]")
    data = _aggregate(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    _save_table(data, config.output_directory / "training_progress.csv")
    figure = _plot(data)
    figure.savefig(
        config.output_directory / "training_progress.svg", bbox_inches="tight"
    )
    figure.savefig(
        config.output_directory / "training_progress.png",
        bbox_inches="tight",
        dpi=config.dpi,
    )
    plt.close(figure)
    print(
        json.dumps(
            {
                "iterations": len(data["iteration"]),
                "maximum_environment_steps": float(data["environment_steps"][-1]),
                "output_directory": str(config.output_directory),
                "seeds": list(config.seeds),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run_hydra(Config, main)
