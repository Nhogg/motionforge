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
        logged_images = []
        logger.log_image = lambda **kwargs: logged_images.append(kwargs)
        callback(0, parameters)
        checks = {
            f"{role}_video": (output / "videos" / f"{role}_000000000000.mp4").is_file()
            for role in ("pursuer", "evader")
        }
        checks["trajectory_images_logged_at_video_step"] = (
            len(logged_images) == 2
            and {item["key"] for item in logged_images}
            == {"evaluation/pursuer_trajectory", "evaluation/evader_trajectory"}
            and all(
                item["environment_steps"] == 0 and item["path"].is_file()
                for item in logged_images
            )
        )
        metadata = {
            role: json.loads(
                (output / "videos" / f"{role}_000000000000.json").read_text()
            )
            for role in ("pursuer", "evader")
        }
        checks["physical_roles_and_colors_stay_fixed"] = all(
            item["agent_roles"] == ["pursuer", "evader"]
            and item["agent_role_encodings"] == [[1.0, 0.0], [0.0, 1.0]]
            and item["opponent_id"] == "current/self-play"
            and item["opponent_role"]
            == ("evader" if item["learner_index"] == 0 else "pursuer")
            and item["selected_strategy_heads"] == ["pursuer_head", "evader_head"]
            and item["agent_policy_bindings"]
            == [
                {
                    "physical_agent_index": 0,
                    "role": "pursuer",
                    "strategy_head": "pursuer_head",
                    "parameter_source": "learner"
                    if item["learner_index"] == 0
                    else "opponent",
                },
                {
                    "physical_agent_index": 1,
                    "role": "evader",
                    "strategy_head": "evader_head",
                    "parameter_source": "learner"
                    if item["learner_index"] == 1
                    else "opponent",
                },
            ]
            and item["colors"] == {"green": "pursuer", "orange": "evader"}
            and item["role_trace_validated"]
            and item["role_trace_steps"] == 3
            and all(
                trace == [[1.0, 0.0], [0.0, 1.0]]
                for trace in item["agent_role_trace"]
            )
            and item["roles_validated"]
            for item in metadata.values()
        )
        checks["learner_selection_matches_video_role"] = (
            metadata["pursuer"]["learner_index"] == 0
            and metadata["evader"]["learner_index"] == 1
        )
        checks["all_agent_visual_geoms_colored"] = all(
            min(item["colored_geom_counts"]) > 10 for item in metadata.values()
        )
        checks["plots_use_the_video_rollout_samples"] = all(
            len(item["trajectory"]["time_seconds"]) == 3
            and len(item["trajectory"]["positions_world_xy"]) == 3
            for item in metadata.values()
        )
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
