"""Exercise truncated/padded single and balanced PPO rollouts on actual Warp."""

import json
from pathlib import Path

import jax
import numpy as np

from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.envs.tag_endings import TagEndCause
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
)
from motionforge.training import (
    HierarchicalPpoTrainer,
    HierarchicalPpoUpdater,
    HierarchicalTrainerConfig,
    HierarchicalUpdateConfig,
    hierarchical_tag_training_metrics,
)


def main():
    _, p2 = load_p2_checkpoint(
        Path(
            "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
        )
    )
    env = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None)),
        config=MultiAgentTaskConfig(episode_duration=0.06, naconmax=128, njmax=4096),
    )
    runner = HierarchicalRolloutRunner(
        env,
        p2_locomotion_normalization(p2[0]),
        HierarchicalRolloutConfig(stochastic_actions=True),
    )
    initial = runner.reset(jax.random.PRNGKey(0))
    params = migrate_p2_actor_parameters(
        runner.initialize_parameters(jax.random.PRNGKey(1), initial), p2[1]
    )
    env.league = FixedLeague(LeagueOpponent("historical/test", "historical", params))
    updater = HierarchicalPpoUpdater(
        HierarchicalUpdateConfig(freeze_locomotion=True, minibatch_size=2)
    )
    state = updater.initialize(params)
    trainer = HierarchicalPpoTrainer(
        runner,
        updater,
        HierarchicalTrainerConfig(
            rollout_horizon=10,
            strategy_advantage=AdvantageConfig(discount=1.0, gae_lambda=1.0),
        ),
    )
    _, final_single, single = trainer.train_iteration(
        state, initial, seed=0, episode_id=0, environment_id=0
    )
    initial_batch = jax.vmap(runner.reset)(jax.random.split(jax.random.PRNGKey(10), 2))
    print(
        "reset buffer shapes",
        initial_batch.environment.physics.data.qpos.shape,
        initial_batch.environment.physics.data._impl.contact__dim.shape,
        flush=True,
    )
    segment_debug = runner.start_segment(params, seed=0, learner_is_pursuer=True)
    stepped_debug = jax.vmap(lambda one: runner.step(segment_debug, one))(initial_batch)
    print(
        "step buffer shapes",
        stepped_debug.environment.physics.data.qpos.shape,
        stepped_debug.environment.physics.data._impl.contact__dim.shape,
        flush=True,
    )
    trained, final, result = trainer.train_batch_iteration(
        state, initial_batch, seed=0, episode_id=0
    )
    final.environment.physics.data.qpos.block_until_ready()
    metrics = {
        k: float(np.asarray(v))
        for k, v in hierarchical_tag_training_metrics(result.batch).items()
    }
    checks = {
        "exclusive_truncation": bool(
            np.asarray(final.environment.truncated).all()
            and not np.asarray(final.environment.terminated).any()
        ),
        "unique_final_cause": bool(
            np.all(
                np.asarray(final.environment.termination.cause)
                == TagEndCause.TIME_LIMIT_TRUNCATION
            )
        ),
        "final_observation_not_padding": bool(
            np.all(np.asarray(final.locomotion_steps) == 3)
            and int(final_single.locomotion_steps) == 3
        ),
        "validity_mask": np.array_equal(
            np.asarray(result.batch.transition_valid).reshape(2, 10),
            np.asarray([[True] * 3 + [False] * 7] * 2),
        ),
        "zero_tag_component_on_timeout": bool(
            np.all(
                np.asarray(result.batch.transitions.strategy_reward_components.tag) == 0
            )
        ),
        "potential_telescope_at_strategy_cadence": all(
            np.isclose(
                metrics[f"reward/{role}/progress_episode_sum"],
                (1.0 if role == "pursuer" else -1.0)
                * 0.25
                * (
                    metrics[f"{role}/initial_distance"]
                    - metrics[f"{role}/final_distance"]
                )
                / (8.0 * np.sqrt(2.0)),
                atol=1e-6,
            )
            for role in ("pursuer", "evader")
        ),
        "metrics_finite": all(np.isfinite(v) for v in metrics.values()),
        "ending_assertions": metrics["diagnostics/incompatible_end_flags"] == 0
        and metrics["diagnostics/missing_or_multiple_final_causes"] == 0,
        "locomotion_parameters_and_optimizer_frozen": all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree.leaves(state.parameters["locomotion"]),
                jax.tree.leaves(trained.parameters["locomotion"]),
                strict=True,
            )
        )
        and all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree.leaves(state.locomotion_optimizer_state),
                jax.tree.leaves(trained.locomotion_optimizer_state),
                strict=True,
            )
        ),
    }
    for role, learner in ((True, 0), (False, 1)):
        segment = runner.start_segment(params, seed=0, learner_is_pursuer=role)
        values, _ = jax.vmap(
            lambda one, segment=segment: runner.bootstrap_values(segment, one)
        )(final)
        reward = np.asarray(result.batch.strategy.reward).reshape(2, 2)[learner, 0]
        actual = np.asarray(result.strategy_targets.returns).reshape(2, 2)[learner, 0]
        checks[f"{'pursuer' if role else 'evader'}_actual_next_state_bootstrap"] = bool(
            np.isclose(actual, reward + float(values[learner, learner]), atol=1e-5)
        )
    single_segment = runner.start_segment(params, seed=0)
    values, _ = runner.bootstrap_values(single_segment, final_single)
    checks["single_trainer_next_state_bootstrap"] = bool(
        np.isclose(
            float(single.strategy_targets.returns[0]),
            float(single.batch.strategy.reward[0])
            + float(values[single_segment.learner_index]),
            atol=1e-5,
        )
    )
    single_valid = np.asarray(single.batch.transition_valid, dtype=bool)
    single_distance = np.asarray(single.batch.transitions.planar_distance)[
        single_valid
    ][-1]
    single_start = float(single.batch.transitions.previous_distance[0])
    checks["single_trainer_shaping_uses_ppo_discount"] = bool(
        np.isclose(
            np.asarray(single.batch.transitions.strategy_reward_components.progress)[
                single_valid
            ].sum(),
            (1.0 if single_segment.learner_is_pursuer else -1.0)
            * 0.25
            * (single_start - single_distance)
            / (8.0 * np.sqrt(2.0)),
            atol=1e-6,
        )
    )
    print(
        json.dumps(
            {
                "checks": {k: bool(v) for k, v in checks.items()},
                "passed": all(checks.values()),
                "metrics": metrics,
            },
            sort_keys=True,
        )
    )
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
