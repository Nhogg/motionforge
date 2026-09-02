"""train_g1_standing.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Run a reproducible PPO training job for MotionForge's G1 locomotion prior.

    This launcher trains the MotionForge G1 environment using MuJoCo Playground's
    Brax PPO implementation (proximal policy optimization). MotionForge owns
    config, provenance, metrics, and checkpoint locations while Playground owns
    physics, observations, rewards, wrappers, and PPO internals.
"""

from __future__ import annotations

import functools
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
import tyro
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from motionforge.envs.g1_standing import (
    G1StandingJoystick,
    default_config,
)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    standing_probability: float = 0.30

    num_timesteps: int = 100_000
    num_envs: int = 128
    num_eval_envs: int = 16
    episode_length: int = 500
    num_evals: int = 2

    unroll_length: int = 10
    batch_size: int = 32
    num_minibatches: int = 4
    num_updates_per_batch: int = 2

    output_dir: Path = Path("logs/p2/training/g1_standing_smoke_a")
    playground_root: Path = Path("../mujoco_playground")


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


def serialize_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {key: json_value(value) for key, value in sorted(metrics.items())}


def validate_config(config: Config) -> None:

    positive_fields = {
        "num_timesteps": config.num_timesteps,
        "num_envs": config.num_eval_envs,
        "num_eval_envs": config.num_eval_envs,
        "episode_length": config.episode_length,
        "num_evals": config.num_evals,
        "unroll_length": config.unroll_length,
        "batch_size": config.batch_size,
        "num_minibatches": config.num_minibatches,
        "num_updates_per_batch": config.num_updates_per_batch,
    }

    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

        if config.output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists: {config.output_dir}"
            )


def main(config: Config) -> None:
    validate_config(config)

    motionforge_root = Path(__file__).resolve().parents[2]
    playground_root = config.playground_root.resolve()

    output_dir = config.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    checkpoint_dir.mkdir(parents=True)

    environment_config = default_config()
    environment_config.standing_probability = config.standing_probability
    environment_config.impl = "warp"

    # Warp allocates contact capacity across the batched worlds.
    environment_config.naconmax = 8 * max(config.num_envs, config.num_eval_envs)

    environment = G1StandingJoystick(config=environment_config)
    evaluation_environment = G1StandingJoystick(config=environment_config)

    ppo_config = locomotion_params.brax_ppo_config(
        "G1JoystickFlatTerrain",
        "warp",
    )
    ppo_config.num_timesteps = config.num_timesteps
    ppo_config.num_envs = config.num_envs
    ppo_config.episode_length = config.episode_length
    ppo_config.num_evals = config.num_evals
    ppo_config.unroll_length = config.unroll_length
    ppo_config.batch_size = config.batch_size
    ppo_config.num_minibatches = config.num_minibatches
    ppo_config.num_updates_per_batch = config.num_updates_per_batch

    network_config = ppo_config.network_factory
    training_parameters = dict(ppo_config)
    del training_parameters["network_factory"]

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **network_config,
    )

    manifest = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "dependencies": {
            "brax": version("brax"),
            "jax": jax.__version__,
            "mujoco": version("mujoco"),
            "playground": version("playground"),
            "python": platform.python_version(),
            "warp": version("warp-lang"),
        },
        "devices": [str(device) for device in jax.devices()],
        "environment_config": environment_config.to_dict(),
        "experiment": "g1_standing_ppo_training_smoke",
        "motionforge_revision": git_output(
            motionforge_root,
            "rev-parse",
            "HEAD",
        ),
        "motionforge_status_short": git_output(
            motionforge_root,
            "status",
            "--short",
        ).splitlines(),
        "playground_revision": git_output(
            playground_root,
            "rev-parse",
            "HEAD",
        ),
        "playground_status_short": git_output(
            playground_root,
            "status",
            "--short",
        ).splitlines(),
        "ppo_config": {
            key: json_value(value) for key, value in training_parameters.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    progress_records: list[dict[str, Any]] = []
    start_time = time.monotonic()

    def progress(
        step: int,
        metrics: Mapping[str, Any],
    ) -> None:
        record = {
            "elapsed_seconds": time.monotonic() - start_time,
            "metrics": serialize_metrics(metrics),
            "step": int(step),
        }
        progress_records.append(record)

        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

        episode_reward = record["metrics"].get("eval/episode_reward")
        print(
            json.dumps(
                {
                    "elapsed_seconds": record["elapsed_seconds"],
                    "episode_reward": episode_reward,
                    "step": record["step"],
                },
                sort_keys=True,
            )
        )

    make_inference_fn, parameters, final_metrics = ppo.train(
        environment=environment,
        eval_env=evaluation_environment,
        network_factory=network_factory,
        seed=config.seed,
        save_checkpoint_path=checkpoint_dir.as_posix(),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=config.num_eval_envs,
        progress_fn=progress,
        policy_params_fn=lambda *_: None,
        vision=False,
        **training_parameters,
    )

    del make_inference_fn, parameters

    elapsed_seconds = time.monotonic() - start_time
    serialized_final_metrics = serialize_metrics(final_metrics)

    checkpoint_entries = sorted(path.name for path in checkpoint_dir.iterdir())

    metric_values = []
    for record in progress_records:
        metric_values.extend(record["metrics"].values())

    finite_metrics = all(
        not isinstance(value, float) or np.isfinite(value) for value in metric_values
    )

    checks = {
        "checkpoint_created": bool(checkpoint_entries),
        "finite_metrics": finite_metrics,
        "gpu_backend": jax.default_backend() == "gpu",
        "progress_recorded": bool(progress_records),
        "training_reached_target": (
            bool(progress_records)
            and progress_records[-1]["step"] >= config.num_timesteps
        ),
    }

    summary = {
        "checks": checks,
        "checkpoint_entries": checkpoint_entries,
        "elapsed_seconds": elapsed_seconds,
        "experiment": "g1_standing_ppo_training_smoke",
        "final_metrics": serialized_final_metrics,
        "metrics_records": len(progress_records),
        "output_dir": str(output_dir),
        "passed": all(checks.values()),
        "seed": config.seed,
        "target_timesteps": config.num_timesteps,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))

    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Config))
