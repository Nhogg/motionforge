"""Audit deterministic P7 policy parity across training and standalone evaluation.

The audit restores one PPO checkpoint, reproduces the training wrapper's reset
key transformation, and verifies that wrapped and standalone observations and
deterministic actions agree exactly. Results are written as machine-readable
JSON so checkpoint-selection semantics remain reproducible.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
from brax.training import checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from motionforge.cli import run_hydra
from motionforge.envs import TagPursuerEnvironment, wrap_tag_pursuer_for_training
from motionforge.policies import make_tag_pursuer_ppo_networks


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p7/training/tag_pursuer_fresh_reset_128k_seed0/checkpoints/"
        "000000081920"
    )
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 1000
    batch_size: int = 2
    output: Path = Path("logs/p7/tag_pursuer_evaluator_parity_a.json")


def main(config: Config) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not config.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {config.checkpoint}")

    environment = TagPursuerEnvironment(
        locomotion_checkpoint=config.locomotion_checkpoint
    )
    wrapped = wrap_tag_pursuer_for_training(
        environment,
        episode_length=environment.tag_environment.timeout_config.maximum_steps,
    )
    input_keys = jax.random.split(jax.random.PRNGKey(config.seed), config.batch_size)
    wrapped_state = jax.jit(wrapped.reset)(input_keys)

    split_keys = jax.vmap(jax.random.split)(input_keys)
    environment_reset_keys = split_keys[:, 0]
    standalone_observations = jax.vmap(environment.reset)(
        environment_reset_keys
    ).obs

    networks = make_tag_pursuer_ppo_networks(running_statistics.normalize)
    parameters = checkpoint.load(config.checkpoint.resolve())
    make_policy = ppo_networks.make_inference_fn(networks)
    deterministic_policy = make_policy(parameters, deterministic=True)
    stochastic_policy = make_policy(parameters, deterministic=False)
    action_keys = jax.random.split(jax.random.PRNGKey(config.seed + 1), config.batch_size)
    wrapped_actions, _ = jax.vmap(deterministic_policy)(
        wrapped_state.obs, action_keys
    )
    standalone_actions, _ = jax.vmap(deterministic_policy)(
        standalone_observations, action_keys
    )
    stochastic_actions, _ = jax.vmap(stochastic_policy)(
        standalone_observations, action_keys
    )

    wrapped_obs = np.asarray(wrapped_state.obs)
    standalone_obs = np.asarray(standalone_observations)
    wrapped_action_values = np.asarray(wrapped_actions)
    standalone_action_values = np.asarray(standalone_actions)
    stochastic_action_values = np.asarray(stochastic_actions)
    checks = {
        "actions_finite": bool(
            np.isfinite(wrapped_action_values).all()
            and np.isfinite(standalone_action_values).all()
        ),
        "deterministic_actions_match": bool(
            np.array_equal(wrapped_action_values, standalone_action_values)
        ),
        "gpu_backend": jax.default_backend() == "gpu",
        "observations_match": bool(np.array_equal(wrapped_obs, standalone_obs)),
        "stochastic_actions_differ": bool(
            not np.array_equal(stochastic_action_values, standalone_action_values)
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "locomotion_checkpoint": str(config.locomotion_checkpoint),
            "output": str(config.output),
        },
        "deterministic_actions": wrapped_action_values.tolist(),
        "experiment": "p7_tag_pursuer_evaluator_parity",
        "jax_version": jax.__version__,
        "observations": wrapped_obs.tolist(),
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "stochastic_actions": stochastic_action_values.tolist(),
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
