"""Train the jointly optimized strategy and locomotion TAG hierarchy."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.league import FixedLeague, LeagueOpponent
from motionforge.logging.wandb import WandbMetricsLogger
from motionforge.models import (
    load_p2_checkpoint,
    migrate_p2_actor_parameters,
    p2_locomotion_normalization,
)
from motionforge.rollout import HierarchicalRolloutConfig, HierarchicalRolloutRunner
from motionforge.training import (
    HierarchicalPpoTrainer,
    HierarchicalPpoUpdater,
    HierarchicalTrainerConfig,
    HierarchicalUpdateConfig,
    load_hierarchical_checkpoint,
    save_hierarchical_checkpoint,
)


@dataclass
class Config:
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    restore_checkpoint: Path | None = None
    seed: int = 0
    num_environments: int = 16
    num_iterations: int = 10
    rollout_horizon: int = 500
    episode_duration: float = 10.0
    checkpoint_interval: int = 5
    naconmax: int = 256
    njmax: int = 512
    strategy_learning_rate: float = 3e-4
    locomotion_learning_rate: float = 1e-5
    strategy_update_interval: int = 1
    locomotion_update_interval: int = 5
    maximum_gradient_norm: float = 1.0
    maximum_locomotion_approximate_kl: float = 0.01
    learner_pursuer_probability: float = 0.5
    boundary_command_safety_enabled: bool = True
    boundary_command_margin: float = 1.0
    wandb_mode: str = "disabled"
    wandb_project: str = "motionforge"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_group: str | None = "hierarchical_tag"
    output_dir: Path = Path("logs/hierarchical/training/tag_joint_smoke_a")


def _git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _json_config(config: Config) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _scalar(value: Any) -> float:
    return float(np.asarray(value))


def _validate_config(config: Config) -> None:
    for name in (
        "num_environments",
        "num_iterations",
        "rollout_horizon",
        "checkpoint_interval",
        "naconmax",
        "njmax",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.rollout_horizon % 5 != 0:
        raise ValueError("rollout_horizon must be divisible by five")
    if config.episode_duration <= 0.0:
        raise ValueError("episode_duration must be positive")
    rollout_duration = config.rollout_horizon * 0.02
    if not math.isclose(rollout_duration, config.episode_duration, abs_tol=1e-9):
        raise ValueError(
            "rollout_horizon must span exactly one episode at the 0.02-second "
            "control timestep"
        )
    if config.num_environments % 2 != 0:
        raise ValueError("num_environments must be even for balanced role batches")
    if not 0.0 <= config.learner_pursuer_probability <= 1.0:
        raise ValueError("learner_pursuer_probability must be in [0, 1]")
    if config.boundary_command_margin <= 0.0:
        raise ValueError("boundary_command_margin must be positive")
    if config.wandb_mode not in {"disabled", "online", "offline"}:
        raise ValueError("wandb_mode must be disabled, online, or offline")
    if not config.locomotion_checkpoint.is_dir():
        raise FileNotFoundError(
            f"locomotion checkpoint does not exist: {config.locomotion_checkpoint}"
        )
    if config.restore_checkpoint is not None and not config.restore_checkpoint.is_dir():
        raise FileNotFoundError(
            f"restore checkpoint does not exist: {config.restore_checkpoint}"
        )
    if config.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {config.output_dir}")


def main(config: Config) -> None:
    _validate_config(config)
    output_dir = config.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    _, p2_parameters = load_p2_checkpoint(config.locomotion_checkpoint)
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None)),
        config=MultiAgentTaskConfig(
            episode_duration=config.episode_duration,
            naconmax=config.naconmax,
            njmax=config.njmax,
        ),
    )
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2_parameters[0]),
        HierarchicalRolloutConfig(
            learner_pursuer_probability=config.learner_pursuer_probability,
            stochastic_actions=True,
            boundary_command_safety_enabled=(config.boundary_command_safety_enabled),
            boundary_command_margin=config.boundary_command_margin,
        ),
    )
    initialization_state = runner.reset(jax.random.PRNGKey(config.seed))
    parameters = migrate_p2_actor_parameters(
        runner.initialize_parameters(
            jax.random.PRNGKey(config.seed + 1), initialization_state
        ),
        p2_parameters[1],
    )
    environment.league = FixedLeague(
        LeagueOpponent(
            "warm-start/p2",
            "warm_start",
            jax.tree.map(lambda value: value, parameters),
        )
    )
    update_config = HierarchicalUpdateConfig(
        strategy_learning_rate=config.strategy_learning_rate,
        locomotion_learning_rate=config.locomotion_learning_rate,
        strategy_update_interval=config.strategy_update_interval,
        locomotion_update_interval=config.locomotion_update_interval,
        maximum_gradient_norm=config.maximum_gradient_norm,
        maximum_locomotion_approximate_kl=(config.maximum_locomotion_approximate_kl),
    )
    updater = HierarchicalPpoUpdater(update_config)
    train_state = updater.initialize(parameters)
    restored_from = None
    if config.restore_checkpoint is not None:
        train_state, restore_manifest = load_hierarchical_checkpoint(
            config.restore_checkpoint, train_state
        )
        if restore_manifest["update_config"] != asdict(update_config):
            raise ValueError("restore checkpoint update configuration does not match")
        restored_from = str(config.restore_checkpoint.resolve())
    trainer = HierarchicalPpoTrainer(
        runner,
        updater,
        HierarchicalTrainerConfig(rollout_horizon=config.rollout_horizon),
    )

    repository = Path(__file__).resolve().parents[2]
    manifest = {
        "config": _json_config(config),
        "dependencies": {
            "jax": jax.__version__,
            "mujoco": version("mujoco"),
            "python": platform.python_version(),
            "warp": version("warp-lang"),
        },
        "devices": [str(device) for device in jax.devices()],
        "experiment": "hierarchical_joint_tag_ppo",
        "motionforge_revision": _git_output(repository, "rev-parse", "HEAD"),
        "motionforge_status_short": _git_output(
            repository, "status", "--short"
        ).splitlines(),
        "p2_locomotion_checkpoint": str(config.locomotion_checkpoint.resolve()),
        "restore_checkpoint": restored_from,
        "update_config": asdict(update_config),
    }
    logger = WandbMetricsLogger(
        mode=config.wandb_mode,
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name or output_dir.name,
        group=config.wandb_group,
        output_dir=output_dir,
        configuration=manifest,
    )
    manifest["wandb"] = logger.metadata
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    start_time = time.monotonic()
    checkpoint_entries: list[str] = []
    records: list[dict[str, Any]] = []
    start_iteration = int(train_state.iteration)
    for local_iteration in range(config.num_iterations):
        iteration = start_iteration + local_iteration
        reset_key = jax.random.fold_in(jax.random.PRNGKey(config.seed), iteration)
        rollout_states = jax.vmap(runner.reset)(
            jax.random.split(reset_key, config.num_environments)
        )
        train_state, final_rollout_states, iteration_result = (
            trainer.train_batch_iteration(
                train_state,
                rollout_states,
                seed=config.seed + iteration,
                episode_id=iteration,
            )
        )
        final_rollout_states.environment.physics.data.qpos.block_until_ready()
        transition_valid = iteration_result.batch.transition_valid
        pursuer_transition = iteration_result.batch.transitions.role[:, 0] > 0.5
        evader_transition = iteration_result.batch.transitions.role[:, 1] > 0.5
        pursuer_tags = (
            transition_valid
            & pursuer_transition
            & iteration_result.batch.transitions.tagged
        )
        evader_timeouts = (
            transition_valid
            & evader_transition
            & iteration_result.batch.transitions.timed_out
        )
        role_environment_count = config.num_environments // 2
        metrics = {
            "locomotion/approximate_kl": _scalar(
                iteration_result.update_metrics.locomotion.approximate_kl
            ),
            "locomotion/entropy": _scalar(
                iteration_result.update_metrics.locomotion.entropy
            ),
            "locomotion/gradient_norm": _scalar(
                iteration_result.update_metrics.locomotion_gradient_norm
            ),
            "locomotion/reward_mean": _scalar(
                iteration_result.batch.locomotion.reward.mean()
            ),
            "locomotion/total_loss": _scalar(
                iteration_result.update_metrics.locomotion.total_loss
            ),
            "strategy/approximate_kl": _scalar(
                iteration_result.update_metrics.strategy.approximate_kl
            ),
            "strategy/entropy": _scalar(
                iteration_result.update_metrics.strategy.entropy
            ),
            "strategy/gradient_norm": _scalar(
                iteration_result.update_metrics.strategy_gradient_norm
            ),
            "strategy/reward_mean": _scalar(
                iteration_result.batch.strategy.reward.mean()
            ),
            "strategy/pursuer_tag_rate": _scalar(
                pursuer_tags.sum() / role_environment_count
            ),
            "strategy/evader_timeout_rate": _scalar(
                evader_timeouts.sum() / role_environment_count
            ),
            "strategy/total_loss": _scalar(
                iteration_result.update_metrics.strategy.total_loss
            ),
            "training/iteration": int(train_state.iteration),
            "training/pursuer_sample_fraction": _scalar(
                iteration_result.batch.strategy.role[:, 0].mean()
            ),
            "training/locomotion_updates": int(train_state.locomotion_updates),
            "training/locomotion_update_rejected": float(
                iteration_result.update_metrics.locomotion_rejected
            ),
            "training/strategy_updates": int(train_state.strategy_updates),
        }
        environment_steps = (
            int(train_state.iteration)
            * config.rollout_horizon
            * config.num_environments
        )
        finite = all(np.isfinite(value) for value in metrics.values())
        record = {
            "elapsed_seconds": time.monotonic() - start_time,
            "environment_steps": environment_steps,
            "metrics": metrics,
            "metrics_finite": finite,
            "opponent_category": iteration_result.trajectory.opponent_category,
            "opponent_id": iteration_result.trajectory.opponent_id,
        }
        records.append(record)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        logger.log(environment_steps=environment_steps, metrics=metrics)
        print(json.dumps(record, sort_keys=True))

        should_checkpoint = (
            int(train_state.iteration) % config.checkpoint_interval == 0
            or local_iteration == config.num_iterations - 1
        )
        if should_checkpoint:
            name = f"{int(train_state.iteration):012d}"
            save_hierarchical_checkpoint(
                checkpoint_dir / name,
                train_state,
                update_config=update_config,
                metadata={
                    "environment_steps": environment_steps,
                    "seed": config.seed,
                },
            )
            checkpoint_entries.append(name)

    elapsed_seconds = time.monotonic() - start_time
    target_iteration = start_iteration + config.num_iterations
    checks = {
        "checkpoint_created": bool(checkpoint_entries),
        "finite_metrics": bool(records)
        and all(record["metrics_finite"] for record in records),
        "gpu_backend": jax.default_backend() == "gpu",
        "progress_recorded": len(records) == config.num_iterations,
        "training_reached_target": int(train_state.iteration) == target_iteration,
    }
    summary = {
        "checks": checks,
        "checkpoint_entries": checkpoint_entries,
        "elapsed_seconds": elapsed_seconds,
        "experiment": "hierarchical_joint_tag_ppo",
        "final_iteration": int(train_state.iteration),
        "output_dir": str(output_dir),
        "passed": all(checks.values()),
        "seed": config.seed,
        "total_environment_steps": (
            int(train_state.iteration)
            * config.rollout_horizon
            * config.num_environments
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    logger.finish(summary)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
