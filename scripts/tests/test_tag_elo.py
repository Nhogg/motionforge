"""Validate role-biased Elo updates and live TAG progress plots."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from motionforge.callbacks import TagProgressPlotCallback
from motionforge.logging.wandb import WandbMetricsLogger
from motionforge.training import TagEloTracker


def main() -> None:
    tracker = TagEloTracker(baseline_pursuer_win_probability=0.65)
    baseline = tracker.pursuer_win_probability(1_000.0, 1_000.0)
    metrics = tracker.update(
        opponent_id="historical/test",
        learner_pursuer_win_rate=1.0,
        opponent_pursuer_win_rate=0.0,
        pursuer_games=32,
        evader_games=32,
    )
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        logger = WandbMetricsLogger(
            mode="disabled",
            project="test",
            entity=None,
            name="test",
            group=None,
            output_dir=output,
            configuration={},
        )
        plot_metrics = {
            **metrics,
            "strategy/pursuer/episode_return_mean": 1.0,
            "strategy/evader/episode_return_mean": 0.0,
        }
        TagProgressPlotCallback(logger, output)(25_000, plot_metrics)
        plots_exist = all(
            (output / "plots" / f"{role}_training_progress.png").is_file()
            for role in ("pursuer", "evader", "combined")
        )
    checks = {
        "baseline_bias_applied": abs(baseline - 0.65) < 1e-12,
        "pursuer_rating_increased": metrics["elo/pursuer/rating"] > 1_000.0,
        "evader_rating_increased": metrics["elo/evader/rating"] > 1_000.0,
        "role_metrics_bounded": all(
            0.0 <= metrics[f"elo/{role}/{name}"] <= 1.0
            for role in ("pursuer", "evader", "combined")
            for name in ("perceived_skill", "task_difficulty")
        ),
        "plots_created": plots_exist,
    }
    result = {
        "baseline_probability": baseline,
        "checks": checks,
        "experiment": "tag_role_biased_elo",
        "passed": all(checks.values()),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
