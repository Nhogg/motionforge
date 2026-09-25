"""Evaluate a learned P7 pursuer against the frozen scripted evader.

The evaluator restores one deterministic high-level PPO checkpoint, runs unseen
reset seeds through the integrated two-G1 MJX-Warp environment, and writes
per-rollout terminal causes plus aggregate pursuit metrics to JSON.
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
from motionforge.envs import (
    TagEnvironmentConfig,
    TagPursuerConfig,
    TagPursuerEnvironment,
)
from motionforge.policies import make_tag_pursuer_ppo_networks


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p7/training/tag_pursuer_128k_seed0/checkpoints/000000122880"
    )
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 1000
    seeds: int = 32
    episode_duration: float = 20.0
    separation: float = 2.0
    arena_half_extent: float = 4.0
    action_repeat: int = 5
    fixed_noise_std: float | None = None
    naconmax: int = 64
    njmax: int = 256
    output: Path = Path("logs/p7/tag_pursuer_122880_heldout_32.json")


def validate_config(config: Config) -> None:
    if config.seeds <= 0:
        raise ValueError("seeds must be positive")
    if config.episode_duration <= 0.0:
        raise ValueError("episode_duration must be positive")
    if config.separation <= 0.0:
        raise ValueError("separation must be positive")
    if config.arena_half_extent <= 0.0:
        raise ValueError("arena_half_extent must be positive")
    if config.action_repeat <= 0:
        raise ValueError("action_repeat must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("contact capacities must be positive")
    if config.fixed_noise_std is not None and config.fixed_noise_std <= 0.001:
        raise ValueError("fixed_noise_std must exceed 0.001")


def terminal_cause(termination) -> str:
    """Assign one diagnostic label with tag taking precedence on ties."""
    if bool(np.asarray(termination.tagged)):
        return "tagged"
    fallen = np.asarray(termination.fallen)
    out_of_bounds = np.asarray(termination.out_of_bounds)
    if bool(fallen[0]):
        return "pursuer_fall"
    if bool(out_of_bounds[0]):
        return "pursuer_out_of_bounds"
    if bool(fallen[1]):
        return "evader_fall"
    if bool(out_of_bounds[1]):
        return "evader_out_of_bounds"
    if bool(np.asarray(termination.timed_out)):
        return "timed_out"
    return "incomplete"


def resolve_fixed_noise_std(
    checkpoint_path: Path, configured_value: float | None
) -> float | None:
    """Resolve the actor distribution contract from its training manifest."""
    if configured_value is not None:
        return configured_value
    manifest_path = checkpoint_path.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = manifest.get("network", {}).get("fixed_noise_std")
    return None if value is None else float(value)


def main(config: Config) -> None:
    validate_config(config)
    checkpoint_path = config.checkpoint.resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    fixed_noise_std = resolve_fixed_noise_std(
        checkpoint_path, config.fixed_noise_std
    )

    environment = TagPursuerEnvironment(
        locomotion_checkpoint=config.locomotion_checkpoint,
        tag_config=TagEnvironmentConfig(
            episode_duration=config.episode_duration,
            separation=config.separation,
            arena_half_extent=config.arena_half_extent,
            naconmax=config.naconmax,
            njmax=config.njmax,
        ),
        pursuer_config=TagPursuerConfig(action_repeat=config.action_repeat),
    )
    network = make_tag_pursuer_ppo_networks(
        running_statistics.normalize,
        fixed_noise_std=fixed_noise_std,
    )
    parameters = checkpoint.load(checkpoint_path)
    policy = ppo_networks.make_inference_fn(network)(
        parameters,
        deterministic=True,
    )
    reset = jax.jit(environment.reset)

    @jax.jit
    def policy_step(state, rng):
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)
        return environment.step(state, action), rng, action

    maximum_steps = round(
        config.episode_duration
        / (environment.tag_environment.config.control_timestep * config.action_repeat)
    )

    def rollout(rollout_seed: int) -> dict:
        rng = jax.random.PRNGKey(rollout_seed)
        rng, reset_rng = jax.random.split(rng)
        state = reset(reset_rng)
        initial_distance = float(np.asarray(state.pipeline_state.distance))
        minimum_distance = initial_distance
        cumulative_reward = 0.0
        maximum_action_magnitude = 0.0
        finite = True
        steps = 0
        last_action = np.zeros(3, dtype=np.float32)

        for _ in range(maximum_steps):
            state, rng, action = policy_step(state, rng)
            state.pipeline_state.tag_state.data.qpos.block_until_ready()
            steps += 1
            action_host = np.asarray(action)
            last_action = action_host
            distance = float(np.asarray(state.pipeline_state.distance))
            minimum_distance = min(minimum_distance, distance)
            cumulative_reward += float(np.asarray(state.reward))
            maximum_action_magnitude = max(
                maximum_action_magnitude,
                float(np.linalg.norm(action_host)),
            )
            tag_state = state.pipeline_state.tag_state
            finite &= bool(
                np.isfinite(np.asarray(tag_state.data.qpos)).all()
                and np.isfinite(np.asarray(tag_state.data.qvel)).all()
                and np.isfinite(np.asarray(state.obs)).all()
                and np.isfinite(action_host).all()
                and np.isfinite(np.asarray(state.reward))
            )
            if bool(np.asarray(state.done)):
                break

        tag_state = state.pipeline_state.tag_state
        return {
            "completed": bool(np.asarray(state.done)),
            "cumulative_reward": cumulative_reward,
            "final_distance": float(np.asarray(state.pipeline_state.distance)),
            "finite": finite,
            "initial_distance": initial_distance,
            "last_action": last_action.tolist(),
            "maximum_action_magnitude": maximum_action_magnitude,
            "minimum_distance": minimum_distance,
            "seed": rollout_seed,
            "steps": steps,
            "terminal_cause": terminal_cause(tag_state.termination),
            "termination": {
                "fallen": np.asarray(tag_state.termination.fallen).tolist(),
                "out_of_bounds": np.asarray(
                    tag_state.termination.out_of_bounds
                ).tolist(),
                "tagged": bool(np.asarray(tag_state.termination.tagged)),
                "timed_out": bool(np.asarray(tag_state.termination.timed_out)),
            },
        }

    rollouts = [rollout(config.seed + offset) for offset in range(config.seeds)]
    repeated = rollout(config.seed)
    cause_counts = {
        cause: sum(item["terminal_cause"] == cause for item in rollouts)
        for cause in (
            "tagged",
            "pursuer_fall",
            "pursuer_out_of_bounds",
            "evader_fall",
            "evader_out_of_bounds",
            "timed_out",
            "incomplete",
        )
    }
    tag_rate = cause_counts["tagged"] / config.seeds
    checks = {
        "all_rollouts_completed": all(item["completed"] for item in rollouts),
        "all_rollouts_finite": all(item["finite"] for item in rollouts),
        "repeatable_rollout_outcome": (
            repeated["terminal_cause"] == rollouts[0]["terminal_cause"]
        ),
        "frozen_evader_identity": environment.evader.fingerprint
        == "4e973d5c79f8e2e1a0de8b44189a7323cd6d9f9aec5f8985671f0668b3ce91b6",
        "gpu_backend": jax.default_backend() == "gpu",
        "outcomes_accounted_for": sum(cause_counts.values()) == config.seeds,
    }
    result = {
        "backend": jax.default_backend(),
        "checkpoint": str(checkpoint_path),
        "checks": checks,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "locomotion_checkpoint": str(config.locomotion_checkpoint),
            "output": str(config.output),
        },
        "evader_fingerprint": environment.evader.fingerprint,
        "experiment": "p7_tag_pursuer_heldout_evaluation",
        "fixed_noise_std": fixed_noise_std,
        "outcomes": {
            "cause_counts": cause_counts,
            "mean_cumulative_reward": float(
                np.mean([item["cumulative_reward"] for item in rollouts])
            ),
            "mean_episode_steps": float(np.mean([item["steps"] for item in rollouts])),
            "mean_minimum_distance": float(
                np.mean([item["minimum_distance"] for item in rollouts])
            ),
            "tag_rate": tag_rate,
        },
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "rollouts": rollouts,
        "seed": config.seed,
        "seeds": config.seeds,
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
