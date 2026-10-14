"""Role-separated diagnostics for hierarchical TAG training batches."""

from __future__ import annotations

import jax.numpy as jp

from motionforge.envs.tag_endings import TagEndCause


def hierarchical_tag_training_metrics(batch) -> dict[str, jp.ndarray]:
    """Summarize task outcomes, returns, and critic values by learner role."""
    environment_count = int(jp.max(batch.transitions.environment_id)) + 1
    transition_steps = batch.transition_valid.shape[0] // environment_count

    def grouped_transition(value):
        return value.reshape((environment_count, transition_steps) + value.shape[1:])

    transition_valid = grouped_transition(batch.transition_valid)
    roles = grouped_transition(batch.transitions.role)[:, 0]
    pursuer = roles[:, 0] > 0.5
    evader = roles[:, 1] > 0.5

    tagged = jp.any(
        grouped_transition(batch.transitions.tagged) & transition_valid, axis=1
    )
    timed_out = jp.any(
        grouped_transition(batch.transitions.timed_out) & transition_valid, axis=1
    )
    fallen = jp.any(
        grouped_transition(
            getattr(batch.transitions, "persistent_fallen", batch.transitions.fallen)
        )
        & transition_valid[..., None],
        axis=1,
    )
    out_of_bounds = jp.any(
        grouped_transition(batch.transitions.out_of_bounds)
        & transition_valid[..., None],
        axis=1,
    )
    episode_steps = jp.sum(transition_valid, axis=1)

    strategy_samples = batch.strategy.reward.shape[0]
    strategy_steps = strategy_samples // environment_count
    rewards = batch.strategy.reward.reshape((environment_count, strategy_steps))
    values = batch.strategy.value.reshape((environment_count, strategy_steps))
    valid = batch.strategy.valid.reshape((environment_count, strategy_steps))
    episode_return = jp.sum(jp.where(valid, rewards, 0.0), axis=1)

    def role_mean(value, mask):
        return jp.sum(value * mask) / jp.maximum(jp.sum(mask), 1)

    def role_value_mean(mask):
        sample_mask = valid & mask[:, None]
        return jp.sum(jp.where(sample_mask, values, 0.0)) / jp.maximum(
            jp.sum(sample_mask), 1
        )

    metrics = {}
    for name, mask, learner_index, opponent_index in (
        ("pursuer", pursuer, 0, 1),
        ("evader", evader, 1, 0),
    ):
        metrics.update(
            {
                f"strategy/{name}/episode_return_mean": role_mean(episode_return, mask),
                f"strategy/{name}/episode_steps_mean": role_mean(episode_steps, mask),
                f"strategy/{name}/fall_rate": role_mean(fallen[:, learner_index], mask),
                f"strategy/{name}/opponent_out_of_bounds_rate": role_mean(
                    out_of_bounds[:, opponent_index], mask
                ),
                f"strategy/{name}/out_of_bounds_rate": role_mean(
                    out_of_bounds[:, learner_index], mask
                ),
                f"strategy/{name}/tag_rate": role_mean(tagged, mask),
                f"strategy/{name}/timeout_rate": role_mean(timed_out, mask),
                f"strategy/{name}/value_mean": role_value_mean(mask),
            }
        )
    if not hasattr(batch.transitions, "terminal_cause"):
        return metrics  # Compatibility with old diagnostic fixtures.

    transitions = batch.transitions
    causes = grouped_transition(transitions.terminal_cause)
    ended = (
        grouped_transition(transitions.terminated | transitions.truncated)
        & transition_valid
    )
    end_count = jp.sum(ended, axis=1)
    final_cause = jp.max(jp.where(ended, causes, 0), axis=1)
    metrics["diagnostics/incompatible_end_flags"] = jp.sum(
        grouped_transition(transitions.terminated & transitions.truncated)
        & transition_valid
    )
    metrics["diagnostics/missing_or_multiple_final_causes"] = jp.sum(end_count != 1)
    learner_indices = grouped_transition(transitions.learner_index)[:, 0]
    previous_distance = grouped_transition(transitions.previous_distance)
    distance = grouped_transition(transitions.planar_distance)
    control_delta = jp.where(transition_valid, previous_distance - distance, 0.0)
    metrics["diagnostics/max_control_distance_jump_m"] = jp.max(jp.abs(control_delta))
    cadence = transition_steps // strategy_steps
    strategy_delta = jp.sum(
        control_delta.reshape(environment_count, strategy_steps, cadence), axis=-1
    )
    metrics["diagnostics/max_strategy_distance_jump_m"] = jp.max(jp.abs(strategy_delta))
    last_indices = jp.maximum(episode_steps - 1, 0)
    rows = jp.arange(environment_count)
    initial_distance = previous_distance[:, 0]
    final_distance = distance[rows, last_indices]
    minimum_distance = jp.minimum(
        initial_distance, jp.min(jp.where(transition_valid, distance, jp.inf), axis=1)
    )
    maximum_distance = jp.maximum(
        initial_distance, jp.max(jp.where(transition_valid, distance, -jp.inf), axis=1)
    )
    dt = grouped_transition(transitions.control_timestep)[:, 0]
    position = grouped_transition(transitions.root_position_world)[
        rows, :, learner_indices, :2
    ]
    extent = grouped_transition(transitions.arena_half_extent)[:, 0]
    boundary_onset = grouped_transition(transitions.boundary_onset_m)[:, 0]
    near_boundary = (
        extent[:, None] - jp.max(jp.abs(position), axis=-1) <= boundary_onset[:, None]
    )

    for name, mask in (("pursuer", pursuer), ("evader", evader)):
        metrics[f"strategy/{name}/valid_tag_games"] = jp.sum(tagged & mask)
        metrics[f"{name}/initial_distance"] = role_mean(initial_distance, mask)
        metrics[f"{name}/final_distance"] = role_mean(final_distance, mask)
        metrics[f"{name}/truncation_rate"] = role_mean(
            final_cause == TagEndCause.TIME_LIMIT_TRUNCATION, mask
        )
        for cause in TagEndCause:
            if cause != TagEndCause.NONE:
                metrics[f"termination/{name}/{cause.name.lower()}_rate"] = role_mean(
                    final_cause == cause, mask
                )
        terms = transitions.strategy_reward_components
        sample_mask = transition_valid & mask[:, None]
        strategy_sample_count = jp.maximum(jp.sum(valid & mask[:, None]), 1)
        for component in (
            "progress",
            "tag",
            "command_smoothness",
            "boundary",
            "fall",
            "oob",
            "total",
        ):
            values_component = grouped_transition(getattr(terms, component))
            sums = jp.sum(jp.where(transition_valid, values_component, 0.0), axis=1)
            metrics[f"reward/{name}/{component}"] = (
                jp.sum(jp.where(sample_mask, values_component, 0.0))
                / strategy_sample_count
            )
            metrics[f"reward/{name}/{component}_episode_sum"] = role_mean(sums, mask)
            metrics[f"reward/{name}/{component}_episode_abs_sum"] = role_mean(
                jp.sum(
                    jp.where(transition_valid, jp.abs(values_component), 0.0), axis=1
                ),
                mask,
            )
            metrics[f"reward/{name}/{component}_episode_abs_max"] = jp.max(
                jp.where(
                    mask,
                    jp.sum(
                        jp.where(transition_valid, jp.abs(values_component), 0.0),
                        axis=1,
                    ),
                    0.0,
                )
            )
        progress_abs_sum = jp.sum(
            jp.where(transition_valid, jp.abs(grouped_transition(terms.progress)), 0.0),
            axis=1,
        )
        metrics[f"reward/{name}/progress_at_least_tag_magnitude_rate"] = role_mean(
            progress_abs_sum >= 1.0, mask
        )
    metrics.update(
        {
            "pursuer/min_distance": role_mean(minimum_distance, pursuer),
            "pursuer/distance_closed": role_mean(
                initial_distance - final_distance, pursuer
            ),
            "pursuer/time_below_1m": role_mean(
                jp.sum((distance < 1.0) & transition_valid, axis=1) * dt, pursuer
            ),
            "pursuer/time_below_0_5m": role_mean(
                jp.sum((distance < 0.5) & transition_valid, axis=1) * dt, pursuer
            ),
            "pursuer/tag_rate": role_mean(tagged, pursuer),
            "pursuer/tag_time_when_successful": role_mean(
                episode_steps * dt, pursuer & tagged
            ),
            "pursuer/tag_success_count": jp.sum(pursuer & tagged),
            "evader/max_distance": role_mean(maximum_distance, evader),
            "evader/separation_gain": role_mean(
                final_distance - initial_distance, evader
            ),
            "evader/time_near_boundary": role_mean(
                jp.sum(near_boundary & transition_valid, axis=1) * dt, evader
            ),
        }
    )
    return metrics
