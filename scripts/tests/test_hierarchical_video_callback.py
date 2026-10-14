"""Smoke-test the in-process hierarchical TAG video callback."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jp

from motionforge.callbacks import HierarchicalTagVideoCallback
from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.league import FixedLeague, LeagueOpponent
from motionforge.logging.wandb import WandbMetricsLogger
from motionforge.models import LocomotionNormalization
from motionforge.rollout import HierarchicalRolloutConfig, HierarchicalRolloutRunner


def main() -> None:
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None)),
        config=MultiAgentTaskConfig(naconmax=32, njmax=256),
    )
    runner = HierarchicalRolloutRunner(
        environment,
        LocomotionNormalization(jp.zeros(103), jp.ones(103)),
        HierarchicalRolloutConfig(stochastic_actions=False),
    )
    state = runner.reset(jax.random.PRNGKey(0))
    parameters = runner.initialize_parameters(jax.random.PRNGKey(1), state)
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
        callback = HierarchicalTagVideoCallback(
            runner=runner,
            logger=logger,
            output_dir=output,
            interval_steps=100,
            duration=0.04,
            fps=10,
            width=320,
            height=240,
        )
        callback(0, parameters)
        checks = {
            f"{role}_video": (output / "videos" / f"{role}_000000000000.mp4").is_file()
            for role in ("pursuer", "evader")
        }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "experiment": "hierarchical_video_callback",
        "passed": all(checks.values()),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
