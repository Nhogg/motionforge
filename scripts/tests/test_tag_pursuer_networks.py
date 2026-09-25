"""Validate the initial separate 128-by-128 TAG actor and critic networks."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks

from motionforge.cli import run_hydra
from motionforge.policies import (
    TAG_PURSUER_ACTION_SIZE,
    TAG_PURSUER_HIDDEN_LAYER_SIZES,
    TAG_PURSUER_OBSERVATION_SIZE,
    make_tag_pursuer_ppo_networks,
)


@dataclass
class Config:
    seed: int = 0
    batch_size: int = 8
    output: Path = Path("logs/p7/tag_pursuer_networks_a.json")


def parameter_count(parameters) -> int:
    return sum(int(value.size) for value in jax.tree.leaves(parameters))


def main(config: Config) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    networks = make_tag_pursuer_ppo_networks()
    fixed_noise_std = 0.2
    fixed_networks = make_tag_pursuer_ppo_networks(
        fixed_noise_std=fixed_noise_std
    )
    policy_key, value_key, action_key = jax.random.split(
        jax.random.PRNGKey(config.seed), 3
    )
    policy_parameters = networks.policy_network.init(policy_key)
    fixed_policy_parameters = fixed_networks.policy_network.init(policy_key)
    value_parameters = networks.value_network.init(value_key)
    observations = jp.zeros(
        (config.batch_size, TAG_PURSUER_OBSERVATION_SIZE), dtype=jp.float32
    )
    policy_logits = networks.policy_network.apply(None, policy_parameters, observations)
    fixed_policy_logits = fixed_networks.policy_network.apply(
        None, fixed_policy_parameters, observations
    )
    values = networks.value_network.apply(None, value_parameters, observations)
    inference = ppo_networks.make_inference_fn(networks)(
        (None, policy_parameters, value_parameters), deterministic=False
    )
    actions, extras = inference(observations, action_key)

    action_host = np.asarray(actions)
    logits_host = np.asarray(policy_logits)
    values_host = np.asarray(values)
    checks = {
        "action_bounded": bool(
            np.all(action_host >= -1.0) and np.all(action_host <= 1.0)
        ),
        "action_finite": bool(np.isfinite(action_host).all()),
        "action_shape": action_host.shape
        == (config.batch_size, TAG_PURSUER_ACTION_SIZE),
        "actor_and_critic_separate": policy_parameters is not value_parameters,
        "hidden_layers": TAG_PURSUER_HIDDEN_LAYER_SIZES == (128, 128),
        "fixed_noise_std": bool(
            np.allclose(
                np.log1p(np.exp(np.asarray(fixed_policy_logits[..., 3:]))) + 0.001,
                fixed_noise_std,
            )
        ),
        "log_prob_shape": np.asarray(extras["log_prob"]).shape == (config.batch_size,),
        "policy_logits_finite": bool(np.isfinite(logits_host).all()),
        "policy_logits_shape": logits_host.shape
        == (config.batch_size, 2 * TAG_PURSUER_ACTION_SIZE),
        "value_finite": bool(np.isfinite(values_host).all()),
        "value_shape": values_host.shape == (config.batch_size,),
    }
    result = {
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "p7_tag_pursuer_network_architecture",
        "jax_version": jax.__version__,
        "network": {
            "action_size": TAG_PURSUER_ACTION_SIZE,
            "activation": "tanh",
            "critic_parameters": parameter_count(value_parameters),
            "distribution": "tanh_normal",
            "hidden_layer_sizes": list(TAG_PURSUER_HIDDEN_LAYER_SIZES),
            "observation_size": TAG_PURSUER_OBSERVATION_SIZE,
            "policy_parameters": parameter_count(policy_parameters),
            "shared_parameters": False,
        },
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "seed": config.seed,
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
