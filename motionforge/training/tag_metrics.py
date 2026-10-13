"""Role-separated diagnostics for hierarchical TAG training batches."""

from __future__ import annotations

import jax.numpy as jp


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
        grouped_transition(batch.transitions.fallen) & transition_valid[..., None],
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
    return metrics
