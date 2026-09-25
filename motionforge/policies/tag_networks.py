"""Neural-network architecture for the initial learned TAG pursuer."""

from __future__ import annotations

import jax
import jax.numpy as jp
from brax.training import networks as brax_networks
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks

TAG_PURSUER_OBSERVATION_SIZE = 9
TAG_PURSUER_ACTION_SIZE = 3
TAG_PURSUER_HIDDEN_LAYER_SIZES = (128, 128)


def make_tag_pursuer_ppo_networks(
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    fixed_noise_std: float | None = None,
) -> ppo_networks.PPONetworks:
    """Build separate two-layer tanh actor and critic networks."""
    if fixed_noise_std is not None and fixed_noise_std <= 0.001:
        raise ValueError("fixed_noise_std must exceed the distribution minimum")
    networks = ppo_networks.make_ppo_networks(
        observation_size=TAG_PURSUER_OBSERVATION_SIZE,
        action_size=TAG_PURSUER_ACTION_SIZE,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=TAG_PURSUER_HIDDEN_LAYER_SIZES,
        value_hidden_layer_sizes=TAG_PURSUER_HIDDEN_LAYER_SIZES,
        activation=jax.nn.tanh,
        distribution_type="tanh_normal",
    )
    if fixed_noise_std is None:
        return networks

    mean_network = brax_networks.make_policy_network(
        param_size=TAG_PURSUER_ACTION_SIZE,
        obs_size=TAG_PURSUER_OBSERVATION_SIZE,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=TAG_PURSUER_HIDDEN_LAYER_SIZES,
        activation=jax.nn.tanh,
        distribution_type="tanh_normal",
    )
    raw_scale = jp.log(jp.expm1(fixed_noise_std - 0.001))

    def apply(processor_params, policy_params, obs):
        mean = mean_network.apply(processor_params, policy_params, obs)
        scale = jp.full_like(mean, raw_scale)
        return jp.concatenate([mean, scale], axis=-1)

    policy_network = brax_networks.FeedForwardNetwork(
        init=mean_network.init,
        apply=apply,
    )
    return networks.replace(policy_network=policy_network)
