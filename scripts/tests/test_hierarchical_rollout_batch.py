"""Validate two-timescale rollout batches and independent GAE targets."""

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
    AdvantageConfig,
    HierarchicalRolloutConfig,
    HierarchicalRolloutRunner,
    collect_hierarchical_rollout,
    generalized_advantage_estimate,
    hierarchical_advantages,
)


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 0
    horizon: int = 10
    output: Path = Path("logs/hierarchical/rollout_batch_a.json")


def main(config: Config) -> None:
    _, p2_parameters = load_p2_checkpoint(config.checkpoint)
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None))
    )
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2_parameters[0]),
        HierarchicalRolloutConfig(learner_pursuer_probability=1.0),
    )
    reset_key, parameter_key = jax.random.split(jax.random.PRNGKey(config.seed))
    state = runner.reset(reset_key)
    learner = migrate_p2_actor_parameters(
        runner.initialize_parameters(parameter_key, state), p2_parameters[1]
    )
    environment.league = FixedLeague(
        LeagueOpponent("historical/batch-test", "historical", learner)
    )
    segment = runner.start_segment(learner, seed=config.seed)

    collect = jax.jit(
        lambda initial: collect_hierarchical_rollout(
            runner,
            segment,
            initial,
            horizon=config.horizon,
            episode_id=jp.asarray(23),
            environment_id=jp.asarray(5),
        )
    )
    final_state, batch = collect(state)
    strategy_targets, locomotion_targets = jax.jit(
        lambda value: hierarchical_advantages(
            batch,
            strategy_bootstrap_value=value,
            locomotion_bootstrap_value=value,
        )
    )(jp.asarray(0.0))
    final_state.environment.physics.data.qpos.block_until_ready()

    synthetic = generalized_advantage_estimate(
        rewards=jp.asarray([1.0, 1.0]),
        values=jp.asarray([0.0, 0.0]),
        dones=jp.asarray([False, True]),
        valid=jp.asarray([True, True]),
        bootstrap_value=jp.asarray(7.0),
        config=AdvantageConfig(discount=1.0, gae_lambda=1.0),
    )
    synthetic_masked = generalized_advantage_estimate(
        rewards=jp.asarray([1.0, 100.0]),
        values=jp.asarray([0.0, 0.0]),
        dones=jp.asarray([True, False]),
        valid=jp.asarray([True, False]),
        bootstrap_value=jp.asarray(7.0),
        config=AdvantageConfig(discount=1.0, gae_lambda=1.0),
    )
    strategy_count = config.horizon // 5
    checks = {
        "advantages_finite": bool(
            np.isfinite(np.asarray(strategy_targets.advantages)).all()
            and np.isfinite(np.asarray(locomotion_targets.advantages)).all()
        ),
        "independent_advantage_shapes": (
            strategy_targets.advantages.shape == (strategy_count,)
            and locomotion_targets.advantages.shape == (config.horizon,)
        ),
        "locomotion_preserves_every_step": (
            batch.locomotion.observation.shape == (config.horizon, 103)
            and batch.locomotion.action.shape == (config.horizon, 29)
        ),
        "rollout_counters": (
            int(np.asarray(final_state.locomotion_steps)) == config.horizon
            and int(np.asarray(final_state.strategy_updates)) == strategy_count
        ),
        "strategy_aggregates_five_steps": (
            batch.strategy.observation.shape == (strategy_count, 11)
            and batch.strategy.command.shape == (strategy_count, 3)
            and np.allclose(
                np.asarray(batch.strategy.reward),
                np.asarray(batch.transitions.strategy_reward).reshape(-1, 5).sum(1),
            )
        ),
        "synthetic_terminal_gae": np.allclose(
            np.asarray(synthetic.advantages), np.asarray([2.0, 1.0])
        ),
        "synthetic_post_terminal_mask": np.allclose(
            np.asarray(synthetic_masked.advantages), np.asarray([1.0, 0.0])
        ),
        "validity_masks": bool(
            np.asarray(batch.transition_valid).all()
            and np.asarray(batch.strategy.valid).all()
            and np.asarray(batch.locomotion.valid).all()
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
        "experiment": "hierarchical_two_timescale_rollout_batch",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "strategy_samples": strategy_count,
        "locomotion_samples": config.horizon,
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
