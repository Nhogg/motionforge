"""Reusable world-space trajectory diagnostics for checkpoint and W&B plots."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from motionforge.evaluation.hierarchical_roles import EVADER_COLOR, PURSUER_COLOR


def trajectory_diagnostics(positions, dt, audit):
    positions = np.asarray(positions)
    if positions.ndim != 3 or positions.shape[1:] != (2, 2) or len(positions) < 2:
        raise ValueError(
            "require at least two samples of both agents' world xy positions"
        )
    if dt <= 0:
        raise ValueError("dt must be positive")
    p, e = audit["pursuer_index"], audit["evader_index"]
    relative = positions[:, e] - positions[:, p]
    distance = np.linalg.norm(relative, axis=-1)
    direction = relative[:-1] / np.maximum(distance[:-1, None], 1e-8)
    velocity = np.diff(positions, axis=0) / dt
    approach = np.sum(velocity[:, p] * direction, axis=-1)
    escape = np.sum(velocity[:, e] * direction, axis=-1)
    return {
        "duration_seconds": (len(positions) - 1) * dt,
        "initial_distance": float(distance[0]),
        "final_distance": float(distance[-1]),
        "mean_pursuer_toward_evader_mps": float(np.mean(approach)),
        "pursuer_moving_away_fraction": float(np.mean(approach < -0.05)),
        "trajectory": {
            "time_seconds": (np.arange(len(positions)) * dt).tolist(),
            "positions_world_xy": positions.tolist(),
            "distance": distance.tolist(),
            "radial_motion": {
                "pursuer_toward_evader": approach.tolist(),
                "evader_away_from_pursuer": escape.tolist(),
            },
        },
    }


def plot_path(payload: dict, output: Path) -> None:
    time = np.asarray(payload["trajectory"]["time_seconds"])
    positions = np.asarray(payload["trajectory"]["positions_world_xy"])
    distance = np.asarray(payload["trajectory"]["distance"])
    p, e = payload["pursuer_index"], payload["evader_index"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    extent = payload["arena_half_extent"]
    axes[0].add_patch(
        plt.Rectangle(
            (-extent, -extent),
            2 * extent,
            2 * extent,
            fill=False,
            linestyle="--",
            color="0.65",
        )
    )
    for agent, name, color in (
        (p, "Pursuer", PURSUER_COLOR),
        (e, "Evader", EVADER_COLOR),
    ):
        path = positions[:, agent]
        axes[0].plot(
            path[:, 0], path[:, 1], color=color, label=f"{name} (agent {agent})"
        )
        axes[0].scatter(*path[0], color=color, marker="o", s=65)
        axes[0].scatter(*path[-1], color=color, marker="X", s=75)
        for index in np.unique(np.linspace(0, len(path) - 2, 6).astype(int)):
            stop = min(index + max(1, len(path) // 20), len(path) - 1)
            axes[0].annotate(
                "",
                path[stop],
                path[index],
                arrowprops={"arrowstyle": "->", "color": color},
            )
        axes[0].annotate(
            f"{name} start",
            path[0],
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
        )
        axes[0].annotate(
            f"{time[-1]:.2f}s",
            path[-1],
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set(
        xlabel="World x (m)",
        ylabel="World y (m)",
        title="Agent paths: circles=start, X=end",
        aspect="equal",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(time, distance, color="0.25")
    axes[1].set(
        xlabel="Time (s)", ylabel="Separation (m)", title="Distance between agents"
    )
    radial = payload["trajectory"]["radial_motion"]
    axes[2].plot(
        time[1:],
        radial["pursuer_toward_evader"],
        color=PURSUER_COLOR,
        label="Pursuer toward evader (+)",
    )
    axes[2].plot(
        time[1:],
        radial["evader_away_from_pursuer"],
        color=EVADER_COLOR,
        label="Evader away from pursuer (+)",
    )
    axes[2].axhline(0, color="0.5", linewidth=0.8)
    axes[2].set(
        xlabel="Time (s)",
        ylabel="Radial speed (m/s)",
        title="Actual movement relative to opponent",
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    mode = "stochastic" if payload["stochastic_actions"] else "deterministic"
    name = payload.get("run_name") or Path(payload["checkpoint"]).parent.parent.name
    point = (
        f"step {payload['environment_steps']:,}"
        if "environment_steps" in payload
        else f"iter {payload['checkpoint_iteration']}"
    )
    fig.suptitle(
        f"{name} / {point}\n"
        f"{payload['opponent_id']} | {mode} | seed {payload['seed']} | tag={payload['termination']['tagged']}",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
