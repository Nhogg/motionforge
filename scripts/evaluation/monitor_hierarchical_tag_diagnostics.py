"""Summarize shared training logs without consuming a GPU or querying Slurm."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A writer may still be appending the final record.
            continue
    return result


def snapshot(root: Path, job: str) -> dict:
    now = datetime.now(ZoneInfo("America/Chicago"))
    runs = []
    for directory in sorted(
        (root / "logs/hierarchical/training").glob(f"tag_pursuit_diag_*_job{job}")
    ):
        history = records(directory / "metrics.jsonl")
        latest = history[-1] if history else {}
        metrics = latest.get("metrics", {})
        window = history[-20:]
        averages = {
            key: mean(item["metrics"][key] for item in window)
            for key in (
                "strategy/pursuer/tag_rate",
                "strategy/pursuer/out_of_bounds_rate",
                "strategy/evader/timeout_rate",
            )
            if window
        }
        evaluations = []
        for path in sorted((directory / "evaluation").glob("*.json")):
            try:
                values = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            evaluations.append(
                {
                    "file": path.name,
                    "checks_passed": values.get("passed"),
                    "roles": [
                        {
                            key: scenario[key]
                            for key in (
                                "role",
                                "role_success_rate",
                                "tag_rate",
                                "physical_survival_rate",
                            )
                        }
                        for scenario in values.get("scenarios", [])
                    ],
                }
            )
        summary_path = directory / "summary.json"
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            summary = {}
        frozen = "_frozen_" in directory.name
        runs.append(
            {
                "run": directory.name,
                "iteration": metrics.get("training/iteration", 0),
                "steps": latest.get("environment_steps", 0),
                "steps_per_second": metrics.get(
                    "training/environment_steps_per_second", 0
                ),
                "log_age_seconds": round(
                    time.time() - (directory / "metrics.jsonl").stat().st_mtime
                )
                if history
                else None,
                "recent_records": len(window),
                "recent_mean": averages,
                "all_metrics_finite": all(
                    item.get("metrics_finite", False) for item in history
                ),
                "freeze_ok": not frozen
                or metrics.get("training/locomotion_updates", 0) == 0,
                "training_complete": summary.get("passed", False),
                "evaluations": evaluations,
                "evaluations_complete": len(evaluations) == 4,
                "stationary_gate": (
                    averages.get("strategy/pursuer/tag_rate", 0) >= 0.8
                    and averages.get("strategy/pursuer/out_of_bounds_rate", 1) <= 0.05
                )
                if "_stationary_" in directory.name and len(history) >= 100
                else None,
            }
        )
    gates = [
        run["stationary_gate"] for run in runs if run["stationary_gate"] is not None
    ]
    if len(gates) < 4:
        interpretation = (
            "Awaiting complete stationary runs; early batches do not select a recipe."
        )
    elif not any(gates):
        interpretation = "No stationary run meets the readiness gate: inspect rewards, observations, commands and termination before league scaling."
    elif not all(run["evaluations_complete"] for run in runs) or len(runs) < 8:
        interpretation = "Some stationary pursuit works; await matched evaluation probes before comparing recipes."
    else:
        interpretation = "All evaluations are available: compare conditions across both training seeds and both action modes; historical 005/020 are development opponents, not held-out validation."
    errors = []
    for path in sorted((root / "logs/slurm").glob(f"tag-pursuit-diag-{job}_*.err")):
        matches = [
            line
            for line in path.read_text(errors="replace").splitlines()
            if re.search(
                r"Traceback|CUDA_ERROR|out of memory|slurmstepd: error|Killed", line
            )
        ]
        if matches:
            errors.append({"file": path.name, "lines": matches[-5:]})
    return {
        "updated": now.isoformat(),
        "job": job,
        "runs": runs,
        "interpretation": interpretation,
        "errors": errors,
    }


def publish(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Pursuit diagnostics {report['job']}",
        "",
        f"Updated: {report['updated']}",
        "",
        report["interpretation"],
        "",
        "Training rates below average the last 20 records; mixed opponents differ between records.",
        "",
        "| Run | Iteration | Steps | Pursuer tags | Pursuer OOB | Evader timeouts | Frozen weights OK | Evals | Log age (min) |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for run in report["runs"]:
        rates = run["recent_mean"]
        lines.append(
            f"| {run['run']} | {run['iteration']} | {run['steps']:,} | "
            f"{rates.get('strategy/pursuer/tag_rate', 0):.1%} | "
            f"{rates.get('strategy/pursuer/out_of_bounds_rate', 0):.1%} | "
            f"{rates.get('strategy/evader/timeout_rate', 0):.1%} | {run['freeze_ok']} | {len(run['evaluations'])}/4 | {(run['log_age_seconds'] or 0) / 60:.1f} |"
        )
    for error in report["errors"]:
        lines += [
            "",
            f"Error markers in {error['file']}:",
            "```",
            *error["lines"],
            "```",
        ]
    lines += [
        "",
        "Evaluation checks passing validates execution, not task success.",
        "",
        "| Run | Evaluation | Role | Success | Tags | Physical survival |",
        "|---|---|---|---:|---:|---:|",
    ]
    for run in report["runs"]:
        for evaluation in run["evaluations"]:
            for role in evaluation["roles"]:
                lines.append(
                    f"| {run['run']} | {evaluation['file']} | {role['role']} | "
                    f"{role['role_success_rate']:.1%} | {role['tag_rate']:.1%} | {role['physical_survival_rate']:.1%} |"
                )
    for name, contents in (
        ("status.json", json.dumps(report, indent=2) + "\n"),
        ("status.md", "\n".join(lines) + "\n"),
    ):
        temporary = output / (name + ".tmp")
        temporary.write_text(contents)
        os.replace(temporary, output / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--until", type=datetime.fromisoformat)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.until is not None and args.until.tzinfo is None:
        parser.error("--until must include a UTC offset")
    root = Path(__file__).resolve().parents[2]
    output = root / "logs/hierarchical/diagnostics" / f"job{args.job}"
    while True:
        report = snapshot(root, args.job)
        publish(report, output)
        print(
            json.dumps(
                {
                    "updated": report["updated"],
                    "runs": len(report["runs"]),
                    "evaluated": sum(
                        run["evaluations_complete"] for run in report["runs"]
                    ),
                }
            ),
            flush=True,
        )
        if args.until is None:
            break
        remaining = args.until.timestamp() - time.time()
        if remaining <= 0:
            break
        time.sleep(min(args.interval, remaining))


if __name__ == "__main__":
    main()
