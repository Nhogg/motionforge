"""test_g1_checkpoint_restore.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Test restoration and deterministic inference of a trained G1 checkpoint.

    Inputs are a Brax checkpoint directory, seed, and short rollout length. The
    script reconstructs the saved PPO network, performs deterministic inference
    in the MotionForge G1 environment, and writes machine-readable validation
    output. Tests artifact usability.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
import tyro
from brax.training import checkpoint
from brax.training.agents.ppo import networks as ppo_networks

from motionforge.compat.brax_checkpoint import load_ppo_network
from motionforge.envs.g1_standing import G1StandingJoystick, default_config


@dataclass(frozen=True)
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_standing_full_seed0/checkpoints/000191692800"
    )
    seed: int = 0
    rollout_steps: int = 10
    output: Path = Path("logs/p2/g1_checkpoint_restore_best_a.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main(config: Config) -> None:
    if config.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive")

    checkpoint_path = config.checkpoint.resolve()
    network_config_path = checkpoint_path / "ppo_network_config.json"

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_path}"
        )

    if not network_config_path.is_file():
        raise FileNotFoundError(
            f"Network configuration does not exist: {network_config_path}"
        )

    ppo_network = load_ppo_network(network_config_path)
    parameters = checkpoint.load(checkpoint_path)
    policy = ppo_networks.make_inference_fn(ppo_network)(
        parameters,
        deterministic=True,
    )

    environment_config = default_config()
    environment_config.impl = "warp"
    environment_config.naconmax = 8
    environment_config.njmax = 128

    environment = G1StandingJoystick(config=environment_config)

    rng = jax.random.PRNGKey(config.seed)
    rng, reset_rng, action_rng = jax.random.split(rng, 3)
    state = environment.reset(reset_rng)

    first_action, _ = policy(state.obs, action_rng)
    repeated_action, _ = policy(state.obs, action_rng)

    first_action_array = np.asarray(first_action)
    repeated_action_array = np.asarray(repeated_action)

    actions_finite = bool(np.isfinite(first_action_array).all())
    deterministic_action = bool(
        np.array_equal(first_action_array, repeated_action_array)
    )

    total_reward = 0.0
    state_finite = True
    steps_executed = 0
    episode_ended = False

    step_policy = jax.jit(
        lambda current_state, key: environment.step(
            current_state,
            policy(current_state.obs, key)[0],
        )
    )

    for _ in range(config.rollout_steps):
        rng, action_rng = jax.random.split(rng)
        state = step_policy(state, action_rng)

        qpos = np.asarray(state.data.qpos)
        qvel = np.asarray(state.data.qvel)
        reward = float(np.asarray(state.reward))
        done = bool(np.asarray(state.done))

        state_finite &= bool(
            np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(reward)
        )
        total_reward += reward
        steps_executed += 1

        if done:
            episode_ended = True
            break

    checks = {
        "action_shape": list(first_action_array.shape) == [29],
        "actions_finite": actions_finite,
        "checkpoint_loaded": len(parameters) == 3,
        "deterministic_action": deterministic_action,
        "gpu_backend": jax.default_backend() == "gpu",
        "rollout_completed": steps_executed == config.rollout_steps,
        "state_finite": state_finite,
    }

    result = {
        "action_shape": list(first_action_array.shape),
        "backend": jax.default_backend(),
        "checkpoint": str(checkpoint_path),
        "checks": checks,
        "episode_ended": episode_ended,
        "experiment": "g1_checkpoint_restore_smoke",
        "jax_version": jax.__version__,
        "mujoco_version": version("mujoco"),
        "network_config_sha256": sha256_file(network_config_path),
        "observation_shapes": {
            key: list(np.asarray(value).shape) for key, value in state.obs.items()
        },
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "rollout_steps": config.rollout_steps,
        "seed": config.seed,
        "steps_executed": steps_executed,
        "total_reward": total_reward,
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
    main(tyro.cli(Config))
