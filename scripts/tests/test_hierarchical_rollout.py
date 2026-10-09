"""Exercise 500/50/10 Hz hierarchical execution with P2 locomotion."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskEnv
from motionforge.league import FixedLeague, LeagueOpponent
from motionforge.models import (
    load_p2_checkpoint,
    migrate_p2_actor_parameters,
    p2_locomotion_normalization,
)
from motionforge.rollout import HierarchicalRolloutConfig, HierarchicalRolloutRunner


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 0
    locomotion_steps: int = 10
    stochastic_actions: bool = False
    output: Path = Path("logs/hierarchical/rollout_rates_a.json")


def main(config: Config) -> None:
    if config.locomotion_steps < 6:
        raise ValueError("locomotion_steps must be at least six")
    _, p2_parameters = load_p2_checkpoint(config.checkpoint)
    placeholder_league = FixedLeague(
        LeagueOpponent(
            opponent_id="initialization-placeholder",
            category="scripted",
            policy=None,
        )
    )
    environment = MultiAgentTaskEnv(league=placeholder_league)
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2_parameters[0]),
        HierarchicalRolloutConfig(
            learner_pursuer_probability=1.0,
            stochastic_actions=config.stochastic_actions,
        ),
    )
    reset_key, parameter_key = jax.random.split(jax.random.PRNGKey(config.seed))
    state = runner.reset(reset_key)
    parameters = runner.initialize_parameters(parameter_key, state)
    parameters = migrate_p2_actor_parameters(parameters, p2_parameters[1])
    opponent_parameters = jax.tree.map(lambda value: value, parameters)
    environment.league = FixedLeague(
        LeagueOpponent(
            opponent_id="historical/test-0001",
            category="historical",
            policy=opponent_parameters,
        )
    )
    segment = runner.start_segment(
        parameters,
        seed=config.seed,
    )
    step = jax.jit(runner.step)

    command_history = []
    action_history = []
    for _ in range(config.locomotion_steps):
        state = step(segment, state)
        command_history.append(np.asarray(state.commands))
        action_history.append(np.asarray(state.previous_actions))
    state.environment.physics.data.qpos.block_until_ready()

    commands = np.stack(command_history)
    actions = np.stack(action_history)
    expected_strategy_updates = (
        config.locomotion_steps
        + runner.config.locomotion_steps_per_strategy_step
        - 1
    ) // runner.config.locomotion_steps_per_strategy_step
    checks = {
        "actions_finite": bool(np.isfinite(actions).all()),
        "actions_shape": actions.shape == (config.locomotion_steps, 2, 29),
        "commands_finite": bool(np.isfinite(commands).all()),
        "commands_held_for_five_steps": bool(
            np.array_equal(commands[0], commands[1])
            and np.array_equal(commands[0], commands[4])
        ),
        "environment_not_done": not bool(np.asarray(state.environment.done)),
        "locomotion_step_count": int(np.asarray(state.locomotion_steps))
        == config.locomotion_steps,
        "physics_step_count": int(
            np.asarray(state.environment.physics.step_count)
        )
        == config.locomotion_steps,
        "rates": (
            runner.config.physics_hz == 500
            and runner.config.locomotion_hz == 50
            and runner.config.strategy_hz == 10
            and environment.config.physics_substeps == 10
            and runner.config.locomotion_steps_per_strategy_step == 5
        ),
        "roles_complementary": bool(
            np.array_equal(
                np.asarray(state.roles.sum(axis=0)),
                np.asarray([1.0, 1.0]),
            )
        ),
        "simulation_time": bool(
            np.isclose(
                float(np.asarray(state.environment.physics.data.time)),
                config.locomotion_steps / runner.config.locomotion_hz,
            )
        ),
        "strategy_update_count": int(np.asarray(state.strategy_updates))
        == expected_strategy_updates,
        "policy_statistics_finite": bool(
            np.isfinite(np.asarray(state.strategy_log_probabilities)).all()
            and np.isfinite(np.asarray(state.strategy_entropies)).all()
            and np.isfinite(
                np.asarray(state.locomotion_log_probabilities)
            ).all()
            and np.isfinite(np.asarray(state.locomotion_entropies)).all()
        ),
        "stochastic_log_probabilities_recorded": bool(
            (not config.stochastic_actions)
            or bool(
                np.any(np.asarray(state.strategy_log_probabilities) != 0.0)
                and np.any(
                    np.asarray(state.locomotion_log_probabilities) != 0.0
                )
            )
        ),
        "segment_indices": (
            segment.learner_index == 0 and segment.opponent_index == 1
        ),
        "segment_roles": (
            segment.learner_role == "pursuer"
            and segment.opponent_role == "evader"
        ),
        "segment_opponent_identity": (
            segment.opponent_id == "historical/test-0001"
            and segment.opponent_category == "historical"
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
        "experiment": "hierarchical_multirate_rollout",
        "final_root_heights": np.asarray(
            state.environment.diagnostics.root_height
        ).tolist(),
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "strategy_updates": int(np.asarray(state.strategy_updates)),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
