"""Exercise learner-only hierarchical transition collection on GPU."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskEnv
from motionforge.league import FixedLeague, LeagueOpponent
from motionforge.models import (
    load_p2_checkpoint,
    migrate_p2_actor_parameters,
    p2_locomotion_normalization,
)
from motionforge.rollout import (
    HierarchicalRolloutConfig,
    HierarchicalRolloutRunner,
    collect_hierarchical_transition,
    trajectory_metadata,
)


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 0
    steps: int = 6
    learner_pursuer_probability: float = 1.0
    output: Path = Path("logs/hierarchical/trajectory_a.json")


def main(config: Config) -> None:
    if config.steps < 6:
        raise ValueError("steps must be at least six")
    if config.learner_pursuer_probability not in (0.0, 1.0):
        raise ValueError("test requires a deterministic endpoint role probability")
    _, p2_parameters = load_p2_checkpoint(config.checkpoint)
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None))
    )
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2_parameters[0]),
        HierarchicalRolloutConfig(
            learner_pursuer_probability=config.learner_pursuer_probability
        ),
    )
    reset_key, parameter_key = jax.random.split(jax.random.PRNGKey(config.seed))
    state = runner.reset(reset_key)
    learner = migrate_p2_actor_parameters(
        runner.initialize_parameters(parameter_key, state), p2_parameters[1]
    )
    environment.league = FixedLeague(
        LeagueOpponent("historical/trajectory-test", "historical", learner)
    )
    segment = runner.start_segment(learner, seed=config.seed)
    metadata = trajectory_metadata(segment)

    collect = jax.jit(
        lambda current: collect_hierarchical_transition(
            runner,
            segment,
            current,
            episode_id=jp.asarray(17),
            environment_id=jp.asarray(3),
        )
    )
    transitions = []
    for _ in range(config.steps):
        state, transition = collect(state)
        transitions.append(transition)
    state.environment.physics.data.qpos.block_until_ready()
    trajectory = jax.tree.map(lambda *values: jp.stack(values), *transitions)

    leaves = jax.tree.leaves(trajectory)
    checks = {
        "all_transition_values_finite": all(
            bool(np.isfinite(np.asarray(value)).all()) for value in leaves
        ),
        "identifiers_recorded": bool(
            np.all(np.asarray(trajectory.episode_id) == 17)
            and np.all(np.asarray(trajectory.environment_id) == 3)
        ),
        "learner_only_policy_shapes": (
            trajectory.strategy_observation.shape == (config.steps, 11)
            and trajectory.strategy_command.shape == (config.steps, 3)
            and trajectory.locomotion_observation.shape == (config.steps, 103)
            and trajectory.locomotion_action.shape == (config.steps, 29)
        ),
        "metadata_records_matchup": (
            metadata.opponent_id == "historical/trajectory-test"
            and metadata.opponent_category == "historical"
            and metadata.learner_role
            == (
                "pursuer"
                if config.learner_pursuer_probability == 1.0
                else "evader"
            )
            and metadata.opponent_role
            == (
                "evader"
                if config.learner_pursuer_probability == 1.0
                else "pursuer"
            )
        ),
        "physical_state_shapes": (
            trajectory.root_position_world.shape == (config.steps, 2, 3)
            and trajectory.root_orientation_wxyz.shape == (config.steps, 2, 4)
            and trajectory.joint_position.shape == (config.steps, 2, 29)
            and trajectory.joint_velocity.shape == (config.steps, 2, 29)
        ),
        "reward_streams_recorded": (
            trajectory.strategy_reward.shape == (config.steps,)
            and trajectory.locomotion_reward.shape == (config.steps,)
            and trajectory.command_tracking_error.shape == (config.steps, 3)
        ),
        "strategy_cadence_recorded": np.array_equal(
            np.asarray(trajectory.strategy_update),
            np.asarray([True, False, False, False, False, True]),
        ),
        "task_state_recorded": (
            trajectory.fallen.shape == (config.steps, 2)
            and trajectory.near_fall.shape == (config.steps, 2)
            and trajectory.relative_position.shape == (config.steps, 2, 2)
            and trajectory.terrain_quaternion_wxyz.shape == (config.steps, 4)
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "output": str(config.output),
        },
        "experiment": "hierarchical_trajectory_collection",
        "jax_version": jax.__version__,
        "metadata": asdict(metadata),
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
