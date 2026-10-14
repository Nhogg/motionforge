"""CPU collector fixtures prove potential telescoping and cutoff masking."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
from flax import struct

from motionforge.envs.multi_agent_task import MultiAgentTaskTermination
from motionforge.envs.tag_endings import TagEndCause
from motionforge.rollout import collect_hierarchical_rollout
from motionforge.rollout.tag_reward import TagRewardConfig, TagRewardTerms
from motionforge.rollout.trajectory import HierarchicalTransition
from motionforge.training import hierarchical_tag_training_metrics


@struct.dataclass
class FixtureEnvironment:
    termination: MultiAgentTaskTermination
    done: object


@struct.dataclass
class FixtureState:
    environment: FixtureEnvironment
    steps: object


def ending(truncated):
    return MultiAgentTaskTermination(
        jp.asarray(False),
        jp.zeros(2, bool),
        jp.zeros(2, bool),
        truncated,
        jp.asarray(False),
        jp.asarray(False),
        truncated,
        jp.where(truncated, TagEndCause.TIME_LIMIT_TRUNCATION, TagEndCause.NONE),
    )


def main():
    config = TagRewardConfig()
    runner = SimpleNamespace(
        config=SimpleNamespace(locomotion_steps_per_strategy_step=5, tag_reward=config),
        environment=SimpleNamespace(config=SimpleNamespace(arena_half_extent=4.0)),
    )
    segment = SimpleNamespace(learner_index=0, learner_is_pursuer=True)
    for cutoff in (3, 100):

        def transition(_runner, _segment, state, cutoff=cutoff, **_kwargs):
            index = state.steps
            truncated = index + 1 >= cutoff
            term = ending(truncated)
            next_state = state.replace(
                environment=FixtureEnvironment(term, truncated), steps=index + 1
            )
            data = {
                field.name: jp.asarray(0.0) for field in fields(HierarchicalTransition)
            }
            current = 2.0 - 0.05 * (index + 1)
            data.update(
                strategy_observation=jp.zeros(11),
                strategy_command=jp.zeros(3),
                strategy_update=index % 5 == 0,
                locomotion_observation=jp.zeros(103),
                locomotion_action=jp.zeros(29),
                command_tracking_error=jp.zeros(3),
                done=truncated,
                role=jp.asarray([1.0, 0.0]),
                learner_index=jp.asarray(0),
                episode_id=jp.asarray(0),
                environment_id=jp.asarray(0),
                root_position_world=jp.zeros((2, 3)),
                relative_position=jp.asarray([[current, 0.0], [-current, 0.0]]),
                fallen=jp.zeros(2, bool),
                persistent_fallen=jp.zeros(2, bool),
                out_of_bounds=jp.zeros(2, bool),
                tagged=jp.asarray(False),
                timed_out=truncated,
                terminated=jp.asarray(False),
                truncated=truncated,
                terminal_cause=term.cause,
                previous_distance=2.0 - 0.05 * index,
                planar_distance=current,
                previous_command=jp.zeros(3),
                strategy_reward_components=TagRewardTerms(*([jp.asarray(0.0)] * 7)),
                control_timestep=jp.asarray(0.02),
                arena_half_extent=jp.asarray(4.0),
                boundary_onset_m=jp.asarray(1.0),
            )
            return next_state, HierarchicalTransition(**data)

        initial = FixtureState(
            FixtureEnvironment(ending(jp.asarray(False)), jp.asarray(False)),
            jp.asarray(0),
        )
        with patch(
            "motionforge.rollout.batch.collect_hierarchical_transition", transition
        ):
            final, batch = jax.jit(
                lambda initial=initial: collect_hierarchical_rollout(
                    runner,
                    segment,
                    initial,
                    horizon=10,
                    episode_id=jp.asarray(0),
                    environment_id=jp.asarray(0),
                    strategy_discount=1.0,
                )
            )()
        scale = 0.25 / (8.0 * np.sqrt(2.0))
        expected = [scale * 0.15, 0.0] if cutoff == 3 else [scale * 0.25, scale * 0.25]
        np.testing.assert_allclose(batch.strategy.reward, expected, atol=1e-7)
        np.testing.assert_allclose(
            batch.transitions.strategy_reward_components.total,
            batch.transitions.strategy_reward,
        )
        assert int(final.steps) == min(cutoff, 10)
        assert bool(final.environment.termination.truncated)
        metrics = {
            k: float(v) for k, v in hierarchical_tag_training_metrics(batch).items()
        }
        assert metrics["diagnostics/missing_or_multiple_final_causes"] == 0
        assert metrics["diagnostics/incompatible_end_flags"] == 0
        assert np.isclose(
            metrics["pursuer/distance_closed"], 0.05 * min(cutoff, 10), atol=1e-6
        )
        assert np.isclose(
            metrics["reward/pursuer/progress_episode_sum"], sum(expected), atol=1e-7
        )
        assert metrics["pursuer/truncation_rate"] == 1
        assert all(np.isfinite(value) for value in metrics.values())
    print(
        "Strategy-cadence potential shaping, cutoff retention, and diagnostics passed"
    )


if __name__ == "__main__":
    main()
