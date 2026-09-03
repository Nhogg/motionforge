"""brax_checkpoint.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Compatibility helpers for restoring Brax 0.14.2 network configurations.
"""

from __future__ import annotations

import json
from pathlib import Path

from brax.training import checkpoint, networks
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

_INITIALIZER_FIELDS = (
    "policy_network_kernel_init_fn",
    "value_network_kernel_init_fn",
    "mean_kernel_init_fn",
)


def load_ppo_network(config_path: Path) -> ppo_networks.PPONetworks:
    """Reconstruct a PPO network from a Brax checkpoint configuration.

    Brax 0.14.2 attempts to look up optional initializer fields even when their
    saved value is null. Removeing null fields restores the intended default
    arg behavior.
    """
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    network_kwargs = loaded["network_factory_kwargs"]

    activation = network_kwargs.get("activation")
    if isinstance(activation, str):
        network_kwargs["activation"] = networks.ACTIVATION[activation]

    for field in _INITIALIZER_FIELDS:
        initializer = network_kwargs.get(field)

        if initializer is None:
            network_kwargs.pop(field, None)
        elif isinstance(initializer, str):
            network_kwargs[field] = networks.KERNEL_INITIALIZER[initializer]

    network_config = config_dict.create(**loaded)
    return checkpoint.get_network(
        network_config,
        ppo_networks.make_ppo_networks,
    )
