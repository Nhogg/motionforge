"""Train the initial P7 high-level pursuer against a frozen scripted evader.

The launcher owns reproducible PPO configuration, provenance, checkpoints,
JSONL metrics, summary output, and optional W&B scalar logging. The learned
policy outputs three velocity commands; opponent logic, locomotion inference,
and two-G1 physics remain separate frozen modules.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import numpy as np
from brax.training.agents.ppo import train as ppo

from motionforge.cli import run_hydra
from motionforge.compat.brax_jax import install_device_put_replicated_adapter
from motionforge.envs import (
    TagEnvironmentConfig,
    TagPursuerConfig,
    TagPursuerEnvironment,
    wrap_tag_pursuer_for_training,
)
from motionforge.logging.wandb import WandbMetricsLogger
from motionforge.policies import make_tag_pursuer_ppo_networks


@dataclass
class Config:
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    num_timesteps: int = 131_072
    num_envs: int = 128
    num_eval_envs: int = 16
    episode_length: int = 200
    num_evals: int = 2
    unroll_length: int = 20
    batch_size: int = 256
    num_minibatches: int = 4
    num_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.99
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    action_repeat: int = 5
    episode_duration: float = 20.0
    separation: float = 2.0
    arena_half_extent: float = 4.0
    naconmax_per_env: int = 64
    njmax: int = 256
    wandb_mode: str = "disabled"
    wandb_project: str = "motionforge"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_group: str | None = "p7_tag_pursuer"
    output_dir: Path = Path("logs/p7/training/tag_pursuer_smoke_a")


def git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def json_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, float) and not np.isfinite(scalar):
            return str(scalar)
    return array.tolist()


def serialize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: json_value(value) for key, value in sorted(metrics.items())}


def metrics_are_finite(metrics: Mapping[str, Any]) -> bool:
    return all(
        not np.issubdtype(np.asarray(value).dtype, np.number)
        or np.isfinite(np.asarray(value)).all()
        for value in metrics.values()
    )


def validate_config(config: Config) -> None:
    positive = {
        "num_timesteps": config.num_timesteps,
        "num_envs": config.num_envs,
        "num_eval_envs": config.num_eval_envs,
        "episode_length": config.episode_length,
        "num_evals": config.num_evals,
        "unroll_length": config.unroll_length,
        "batch_size": config.batch_size,
        "num_minibatches": config.num_minibatches,
        "num_updates_per_batch": config.num_updates_per_batch,
        "learning_rate": config.learning_rate,
        "action_repeat": config.action_repeat,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.wandb_mode not in {"disabled", "online", "offline"}:
        raise ValueError("wandb_mode must be disabled, online, or offline")
    if config.batch_size * config.num_minibatches % config.num_envs != 0:
        raise ValueError("batch_size * num_minibatches must be divisible by num_envs")
    expected_episode_length = round(
        config.episode_duration / (0.02 * config.action_repeat)
    )
    if config.episode_length != expected_episode_length:
        raise ValueError(
            "episode_length must match episode_duration at the high-level rate; "
            f"expected {expected_episode_length}"
        )
    if config.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def main(config: Config) -> None:
    validate_config(config)
    compatibility_installed = install_device_put_replicated_adapter()
    tag_config = TagEnvironmentConfig(
        episode_duration=config.episode_duration,
        separation=config.separation,
        arena_half_extent=config.arena_half_extent,
        naconmax=config.naconmax_per_env * max(config.num_envs, config.num_eval_envs),
        njmax=config.njmax,
    )
    pursuer_config = TagPursuerConfig(action_repeat=config.action_repeat)
    environment = TagPursuerEnvironment(
        locomotion_checkpoint=config.locomotion_checkpoint,
        tag_config=tag_config,
        pursuer_config=pursuer_config,
    )
    evaluation_environment = TagPursuerEnvironment(
        locomotion_checkpoint=config.locomotion_checkpoint,
        tag_config=tag_config,
        pursuer_config=pursuer_config,
    )

    output_dir = config.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    motionforge_root = Path(__file__).resolve().parents[2]
    training_parameters = {
        "action_repeat": 1,
        "batch_size": config.batch_size,
        "bootstrap_on_timeout": True,
        "clipping_epsilon": config.clipping_epsilon,
        "discounting": config.discounting,
        "entropy_cost": config.entropy_cost,
        "episode_length": config.episode_length,
        "gae_lambda": config.gae_lambda,
        "learning_rate": config.learning_rate,
        "max_grad_norm": config.max_grad_norm,
        "normalize_observations": True,
        "num_envs": config.num_envs,
        "num_evals": config.num_evals,
        "num_minibatches": config.num_minibatches,
        "num_timesteps": config.num_timesteps,
        "num_updates_per_batch": config.num_updates_per_batch,
        "reward_scaling": 1.0,
        "unroll_length": config.unroll_length,
    }
    manifest = {
        "compatibility": {"jax_device_put_replicated_adapter": compatibility_installed},
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "dependencies": {
            "brax": version("brax"),
            "jax": jax.__version__,
            "mujoco": version("mujoco"),
            "python": platform.python_version(),
            "warp": version("warp-lang"),
        },
        "devices": [str(device) for device in jax.devices()],
        "evader_fingerprint": environment.evader.fingerprint,
        "evader_specification": environment.evader.specification(),
        "experiment": "p7_tag_pursuer_ppo",
        "locomotion_checkpoint": str(environment.controller.checkpoint_path),
        "motionforge_revision": git_output(motionforge_root, "rev-parse", "HEAD"),
        "motionforge_status_short": git_output(
            motionforge_root, "status", "--short"
        ).splitlines(),
        "network": {
            "activation": "tanh",
            "action_size": 3,
            "hidden_layer_sizes": [128, 128],
            "observation_size": 9,
            "separate_actor_critic": True,
        },
        "ppo": training_parameters,
        "pursuer_config": asdict(pursuer_config),
        "tag_config": asdict(tag_config),
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

    progress_records = []
    start_time = time.monotonic()

    def progress(step: int, metrics: Mapping[str, Any]) -> None:
        record = {
            "elapsed_seconds": time.monotonic() - start_time,
            "metrics": serialize_metrics(metrics),
            "metrics_finite": metrics_are_finite(metrics),
            "step": int(step),
        }
        progress_records.append(record)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        logger.log(environment_steps=record["step"], metrics=record["metrics"])
        print(
            json.dumps(
                {
                    "elapsed_seconds": record["elapsed_seconds"],
                    "episode_reward": record["metrics"].get("eval/episode_reward"),
                    "step": record["step"],
                },
                sort_keys=True,
            )
        )

    def network_factory(
        observation_size: int, action_size: int, preprocess_observations_fn
    ):
        if observation_size not in (environment.observation_size, (9,)):
            raise ValueError(f"unexpected observation size: {observation_size}")
        if action_size != environment.action_size:
            raise ValueError(f"unexpected action size: {action_size}")
        return make_tag_pursuer_ppo_networks(preprocess_observations_fn)

    _, _, final_metrics = ppo.train(
        environment=environment,
        eval_env=evaluation_environment,
        network_factory=network_factory,
        num_eval_envs=config.num_eval_envs,
        progress_fn=progress,
        save_checkpoint_path=checkpoint_dir.as_posix(),
        seed=config.seed,
        vision=False,
        wrap_env_fn=wrap_tag_pursuer_for_training,
        **training_parameters,
    )
    elapsed_seconds = time.monotonic() - start_time
    checkpoint_entries = sorted(path.name for path in checkpoint_dir.iterdir())
    final_metrics = serialize_metrics(final_metrics)
    checks = {
        "checkpoint_created": bool(checkpoint_entries),
        "finite_metrics": all(record["metrics_finite"] for record in progress_records)
        and metrics_are_finite(final_metrics),
        "frozen_evader_identity": environment.evader.fingerprint
        == manifest["evader_fingerprint"],
        "gpu_backend": jax.default_backend() == "gpu",
        "progress_recorded": bool(progress_records),
        "training_reached_target": bool(progress_records)
        and progress_records[-1]["step"] >= config.num_timesteps,
    }
    summary = {
        "checks": checks,
        "checkpoint_entries": checkpoint_entries,
        "elapsed_seconds": elapsed_seconds,
        "experiment": "p7_tag_pursuer_ppo",
        "final_metrics": final_metrics,
        "metrics_records": len(progress_records),
        "output_dir": str(output_dir),
        "passed": all(checks.values()),
        "seed": config.seed,
        "target_timesteps": config.num_timesteps,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    logger.finish(
        {
            "elapsed_seconds": elapsed_seconds,
            "passed": summary["passed"],
        }
    )
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
