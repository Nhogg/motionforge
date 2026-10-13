"""Validate the stationary hierarchical curriculum opponent transform."""

from __future__ import annotations

import json

import jax.numpy as jp
import numpy as np
from flax.core import freeze

from motionforge.training import make_stationary_hierarchical_opponent


def main() -> None:
    command = {
        "kernel": jp.ones((2, 3)),
        "bias": jp.ones((3,)),
    }
    parameters = freeze(
        {
            "strategy": {
                "encoder": {"kernel": jp.ones((2, 2))},
                "pursuer_head": {
                    "command": command,
                    "log_std": jp.zeros((3,)),
                    "value": {"kernel": jp.ones((2, 1))},
                },
                "evader_head": {
                    "command": command,
                    "log_std": jp.zeros((3,)),
                    "value": {"kernel": jp.ones((2, 1))},
                },
            },
            "locomotion": {"kernel": jp.ones((2, 2))},
        }
    )
    stationary = make_stationary_hierarchical_opponent(parameters)
    checks = {
        "source_unchanged": bool(
            np.allclose(parameters["strategy"]["pursuer_head"]["command"]["bias"], 1)
        ),
        "pursuer_command_zero": bool(
            np.allclose(stationary["strategy"]["pursuer_head"]["command"]["kernel"], 0)
        ),
        "evader_command_zero": bool(
            np.allclose(stationary["strategy"]["evader_head"]["command"]["bias"], 0)
        ),
        "strategy_near_deterministic": bool(
            np.allclose(stationary["strategy"]["pursuer_head"]["log_std"], -20)
            and np.allclose(stationary["strategy"]["evader_head"]["log_std"], -20)
        ),
        "locomotion_preserved": bool(
            np.allclose(stationary["locomotion"]["kernel"], 1)
        ),
        "value_preserved": bool(
            np.allclose(stationary["strategy"]["pursuer_head"]["value"]["kernel"], 1)
        ),
    }
    result = {
        "checks": checks,
        "experiment": "stationary_hierarchical_opponent",
        "passed": bool(all(checks.values())),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
