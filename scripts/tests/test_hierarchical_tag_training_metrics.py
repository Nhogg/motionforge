"""Validate role-separated hierarchical TAG training diagnostics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import jax.numpy as jp
import numpy as np

from motionforge.training import hierarchical_tag_training_metrics


def main() -> None:
    roles = jp.asarray(
        [
            [[1.0, 0.0]] * 2,
            [[1.0, 0.0]] * 2,
            [[0.0, 1.0]] * 2,
            [[0.0, 1.0]] * 2,
        ]
    )
    batch = SimpleNamespace(
        transition_valid=jp.ones((4, 2), dtype=bool),
        transitions=SimpleNamespace(
            role=roles,
            tagged=jp.asarray(
                [[False, True], [False, False], [False, True], [False, False]]
            ),
            timed_out=jp.asarray(
                [[False, False], [False, True], [False, False], [False, True]]
            ),
            fallen=jp.asarray(
                [
                    [[False, False], [False, False]],
                    [[False, False], [True, False]],
                    [[False, False], [False, False]],
                    [[False, False], [False, False]],
                ]
            ),
            out_of_bounds=jp.asarray(
                [
                    [[False, False], [False, False]],
                    [[False, False], [False, True]],
                    [[False, False], [False, False]],
                    [[False, False], [False, True]],
                ]
            ),
        ),
        strategy=SimpleNamespace(
            reward=jp.asarray([0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0]),
            value=jp.asarray([0.2, 0.8, 0.1, 0.1, -0.3, -0.8, -0.2, -0.7]),
            valid=jp.ones(8, dtype=bool),
        ),
    )
    metrics = {
        key: float(np.asarray(value))
        for key, value in hierarchical_tag_training_metrics(batch).items()
    }
    checks = {
        "pursuer_tag_rate": np.isclose(metrics["strategy/pursuer/tag_rate"], 0.5),
        "pursuer_return": np.isclose(
            metrics["strategy/pursuer/episode_return_mean"], 0.5
        ),
        "pursuer_opponent_oob": np.isclose(
            metrics["strategy/pursuer/opponent_out_of_bounds_rate"], 0.5
        ),
        "evader_tag_rate": np.isclose(metrics["strategy/evader/tag_rate"], 0.5),
        "evader_oob_rate": np.isclose(
            metrics["strategy/evader/out_of_bounds_rate"], 0.5
        ),
        "evader_return": np.isclose(
            metrics["strategy/evader/episode_return_mean"], -1.0
        ),
        "critic_values_separated": np.isclose(
            metrics["strategy/pursuer/value_mean"], 0.3
        )
        and np.isclose(metrics["strategy/evader/value_mean"], -0.5),
    }
    result = {
        "checks": {key: bool(value) for key, value in checks.items()},
        "experiment": "hierarchical_tag_training_metrics",
        "metrics": metrics,
        "passed": bool(all(checks.values())),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
