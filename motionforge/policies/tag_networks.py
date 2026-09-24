"""Neural-network architecture for the initial learned TAG pursuer."""

from __future__ import annotations

import jax
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks

TAG_PURSUER_OBSERVATION_SIZE = 9
TAG_PURSUER_ACTION_SIZE = 3
TAG_PURSUER_HIDDEN_LAYER_SIZES = (128, 128)


def make_tag_pursuer_ppo_networks(
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
) -> ppo_networks.PPONetworks:
    """Build separate two-layer tanh actor and critic networks."""
    return ppo_networks.make_ppo_networks(
        observation_size=TAG_PURSUER_OBSERVATION_SIZE,
        action_size=TAG_PURSUER_ACTION_SIZE,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=TAG_PURSUER_HIDDEN_LAYER_SIZES,
        value_hidden_layer_sizes=TAG_PURSUER_HIDDEN_LAYER_SIZES,
        activation=jax.nn.tanh,
        distribution_type="tanh_normal",
    )
