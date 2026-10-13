"""Create dependency-free CSV and SVG plots from hierarchical TAG evaluations."""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from motionforge.cli import run_hydra


@dataclass
class Config:
    evaluation_directory: Path = Path("logs/hierarchical/evaluation")
    pattern: str = "*.json"
    output_directory: Path = Path("logs/hierarchical/plots")
    trajectory_pattern: str = "*baseline*.json"


def _scenario(report: dict, role: str) -> dict:
    return next(item for item in report["scenarios"] if item["role"] == role)


def _label(report: dict) -> str:
    opponent = report.get("opponent", {})
    opponent_checkpoint = opponent.get("checkpoint")
    opponent_name = (
        Path(opponent_checkpoint).parents[1].name
        if opponent_checkpoint
        else opponent.get("category", "unknown")
    )
    return f"{report.get('action_mode', 'deterministic')} vs {opponent_name}"


def _rows(reports: list[dict]) -> list[dict[str, object]]:
    rows = []
    for report in reports:
        for role in ("pursuer", "evader"):
            scenario = _scenario(report, role)
            rows.append(
                {
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


def _polyline(
    points: list[tuple[float, float]],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> str:
    xmin, xmax = x_limits
    ymin, ymax = y_limits
    coordinates = []
    for x, y in points:
        px = x0 + (x - xmin) / max(xmax - xmin, 1e-9) * width
        py = y0 + height - (y - ymin) / max(ymax - ymin, 1e-9) * height
        coordinates.append(f"{px:.1f},{py:.1f}")
    return " ".join(coordinates)


def _summary_svg(reports: list[dict]) -> str:
    width, panel_height, margin = 1600, 250, 70
    panels: list[tuple[str, Callable[[dict], float], tuple[float, float]]] = [
        ("Role success rate", lambda row: float(row["role_success_rate"]), (0.0, 1.0)),
        (
            "Physical survival rate",
            lambda row: float(row["physical_survival_rate"]),
            (0.0, 1.0),
        ),
        (
            "Command tracking RMSE",
            lambda row: float(row["mean_command_tracking_rmse"]),
            (0.0, 0.5),
        ),
    ]
    rows = _rows(reports)
    iterations = [float(row["iteration"]) for row in rows]
    xmin, xmax = min(iterations), max(iterations)
    groups: dict[tuple[str, str], list[dict]] = {}
    for report in reports:
        label = _label(report)
        for role in ("pursuer", "evader"):
            scenario = _scenario(report, role)
            groups.setdefault((label, role), []).append(
                {"iteration": report["checkpoint_iteration"], **scenario}
            )
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2"]
    color_by_group = {
        key: colors[index % len(colors)] for index, key in enumerate(groups)
    }
    legend_columns = 2
    legend_rows = (len(groups) + legend_columns - 1) // legend_columns
    total_height = margin + len(panels) * panel_height + legend_rows * 24 + 30
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:system-ui,sans-serif;fill:#111827}.axis{stroke:#6b7280;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.series{fill:none;stroke-width:2.5}</style>",
        '<text x="70" y="35" font-size="22" font-weight="700">Hierarchical TAG evaluation summary</text>',
    ]
    plot_width = width - 2 * margin
    for panel_index, (title, getter, y_limits) in enumerate(panels):
        y0 = margin + panel_index * panel_height
        plot_height = panel_height - 55
        svg.append(
            f'<text x="{margin}" y="{y0 - 12}" font-size="16" font-weight="600">{title}</text>'
        )
        for fraction in (0.0, 0.5, 1.0):
            y = y0 + plot_height * (1.0 - fraction)
            value = y_limits[0] + fraction * (y_limits[1] - y_limits[0])
            svg.append(
                f'<line class="grid" x1="{margin}" y1="{y:.1f}" x2="{width - margin}" y2="{y:.1f}"/>'
            )
            svg.append(
                f'<text x="{margin - 10}" y="{y + 5:.1f}" text-anchor="end" font-size="12">{value:.2f}</text>'
            )
        svg.append(
            f'<line class="axis" x1="{margin}" y1="{y0 + plot_height}" x2="{width - margin}" y2="{y0 + plot_height}"/>'
        )
        svg.append(
            f'<line class="axis" x1="{margin}" y1="{y0}" x2="{margin}" y2="{y0 + plot_height}"/>'
        )
        for key, items in groups.items():
            items = sorted(items, key=lambda item: item["iteration"])
            points = [(float(item["iteration"]), getter(item)) for item in items]
            svg.append(
                f'<polyline class="series" stroke="{color_by_group[key]}" points="{_polyline(points, x0=margin, y0=y0, width=plot_width, height=plot_height, x_limits=(xmin, xmax), y_limits=y_limits)}"/>'
            )
            for x, y in points:
                px = margin + (x - xmin) / max(xmax - xmin, 1e-9) * plot_width
                py = (
                    y0
                    + plot_height
                    - (
                        (y - y_limits[0])
                        / max(y_limits[1] - y_limits[0], 1e-9)
                        * plot_height
                    )
                )
                svg.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" '
                    f'fill="{color_by_group[key]}"/>'
                )
        svg.append(
            f'<text x="{width - margin}" y="{y0 + plot_height + 28}" text-anchor="end" font-size="12">checkpoint iteration</text>'
        )
    legend_y = margin + len(panels) * panel_height - 20
    for index, key in enumerate(groups):
        x = margin + (index % legend_columns) * 740
        y = legend_y + (index // legend_columns) * 24
        svg.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" stroke="{color_by_group[key]}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{x + 32}" y="{y + 5}" font-size="12">{html.escape(key[0])} / {key[1]}</text>'
        )
    svg.append("</svg>")
    return "\n".join(svg)


def _trajectory_svg(report: dict) -> str:
    width, height, margin = 1100, 720, 70
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:system-ui,sans-serif;fill:#111827}.axis{stroke:#6b7280}.grid{stroke:#e5e7eb}.series{fill:none;stroke-width:2}</style>",
        f'<text x="{margin}" y="35" font-size="21" font-weight="700">Iteration {report["checkpoint_iteration"]}: {html.escape(_label(report))}</text>',
    ]
    for role_index, role in enumerate(("pursuer", "evader")):
        scenario = _scenario(report, role)
        x0 = margin + role_index * 520
        y0, plot_width, plot_height = 80, 450, 250
        svg.append(
            f'<text x="{x0}" y="{y0 - 15}" font-size="16" font-weight="600">{role}: distance vs time</text>'
        )
        svg.append(
            f'<rect x="{x0}" y="{y0}" width="{plot_width}" height="{plot_height}" fill="none" class="axis"/>'
        )
        for rollout_index, rollout in enumerate(scenario["rollouts"]):
            trajectory = rollout.get("trajectory")
            if not trajectory:
                continue
            times = trajectory["time_seconds"]
            distances = trajectory["distance"]
            points = list(zip(times, distances, strict=True))
            svg.append(
                f'<polyline class="series" stroke="{colors[rollout_index % len(colors)]}" opacity="0.75" points="{_polyline(points, x0=x0, y0=y0, width=plot_width, height=plot_height, x_limits=(0.0, max(times[-1], 0.1)), y_limits=(0.0, 8.0))}"/>'
            )
        svg.append(
            f'<text x="{x0 + plot_width}" y="{y0 + plot_height + 25}" text-anchor="end" font-size="12">time (s)</text>'
        )
        svg.append(f'<text x="{x0 + 8}" y="{y0 + 18}" font-size="12">8 m</text>')

        y1 = 405
        svg.append(
            f'<text x="{x0}" y="{y1 - 15}" font-size="16" font-weight="600">seed 0 planar trajectories</text>'
        )
        svg.append(
            f'<rect x="{x0}" y="{y1}" width="{plot_width}" height="{plot_height}" fill="none" class="axis"/>'
        )
        trajectory = scenario["rollouts"][0].get("trajectory")
        if trajectory:
            for name, color in (
                ("learner_position", "#2563eb"),
                ("opponent_position", "#dc2626"),
            ):
                positions = trajectory[name]
                points = [(float(point[0]), float(point[1])) for point in positions]
                svg.append(
                    f'<polyline class="series" stroke="{color}" points="{_polyline(points, x0=x0, y0=y1, width=plot_width, height=plot_height, x_limits=(-5.0, 5.0), y_limits=(-5.0, 5.0))}"/>'
                )
        svg.append(
            f'<text x="{x0 + 10}" y="{y1 + 22}" font-size="12" fill="#2563eb">learner</text>'
        )
        svg.append(
            f'<text x="{x0 + 75}" y="{y1 + 22}" font-size="12" fill="#dc2626">opponent</text>'
        )
    svg.append("</svg>")
    return "\n".join(svg)


def main(config: Config) -> None:
    paths = sorted(config.evaluation_directory.glob(config.pattern))
    reports = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") == "hierarchical_tag_checkpoint_evaluation":
            reports.append(payload)
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
    (config.output_directory / "summary.svg").write_text(
        _summary_svg(reports) + "\n", encoding="utf-8"
    )
    trajectory_paths = sorted(
        config.evaluation_directory.glob(config.trajectory_pattern)
    )
    trajectory_count = 0
    for path in trajectory_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("experiment") != "hierarchical_tag_checkpoint_evaluation":
            continue
        destination = config.output_directory / f"{path.stem}_trajectories.svg"
        destination.write_text(_trajectory_svg(report) + "\n", encoding="utf-8")
        trajectory_count += 1
    result = {
        "evaluation_count": len(reports),
        "output_directory": str(config.output_directory),
        "summary_csv": str(config.output_directory / "summary.csv"),
        "summary_svg": str(config.output_directory / "summary.svg"),
        "trajectory_plot_count": trajectory_count,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
