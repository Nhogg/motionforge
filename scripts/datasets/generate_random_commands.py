"""Generate the P6 ordinary random-command locomotion dataset.

Inputs are a deterministic locomotion checkpoint, an initial seed, and an exact
agent-control-step budget. Outputs are JSONL timestep records and a JSON
manifest. The generator uses flat terrain, disables pushes, and leaves the
canonical structured command sampler unchanged.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from motionforge.cli import run_hydra
from motionforge.controllers import G1LocomotionController
from motionforge.datasets import (
    LOCOMOTION_DATASET_SCHEMA_VERSION,
    JsonlDatasetWriter,
    extract_g1_locomotion_sample,
    json_default,
    locomotion_record,
)
from motionforge.envs.g1_standing import G1StandingJoystick, default_config


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    samples: int = 1_024
    naconmax: int = 16
    njmax: int = 128
    fall_minimum_root_height: float = 0.45
    fall_minimum_up_alignment: float = 0.50
    near_fall_minimum_root_height: float = 0.60
    near_fall_minimum_up_alignment: float = 0.80
    output: Path = Path("logs/p6/random_commands_smoke.jsonl")
    manifest: Path = Path("logs/p6/random_commands_smoke_manifest.json")


def validate_config(config: Config) -> None:
    """Reject invalid budgets, thresholds, and output collisions."""
    if config.samples <= 0:
        raise ValueError("samples must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("contact capacities must be positive")
    if config.fall_minimum_root_height <= 0.0:
        raise ValueError("fall_minimum_root_height must be positive")
    if config.near_fall_minimum_root_height < config.fall_minimum_root_height:
        raise ValueError("near-fall height must not be below fall height")
    if not 0.0 <= config.fall_minimum_up_alignment <= 1.0:
        raise ValueError("fall_minimum_up_alignment must be in [0, 1]")
    if (
        not config.fall_minimum_up_alignment
        <= config.near_fall_minimum_up_alignment
        <= 1.0
    ):
        raise ValueError("near-fall alignment must be in [fall threshold, 1]")
    if config.output.resolve() == config.manifest.resolve():
        raise ValueError("output and manifest paths must differ")


def main(config: Config) -> None:
    validate_config(config)
    environment_config = default_config()
    environment_config.impl = "warp"
    environment_config.naconmax = config.naconmax
    environment_config.njmax = config.njmax
    environment_config.push_config.enable = False
    environment = G1StandingJoystick(config=environment_config)
    controller = G1LocomotionController.from_checkpoint(
        config.checkpoint,
        environment,
        deterministic=True,
    )

    reset = jax.jit(environment.reset)
    act = jax.jit(controller.act)
    step = jax.jit(environment.step)
    extract = jax.jit(
        lambda state, command, action, targets: extract_g1_locomotion_sample(
            environment,
            state,
            command,
            action,
            targets,
            fall_minimum_root_height=config.fall_minimum_root_height,
            fall_minimum_up_alignment=config.fall_minimum_up_alignment,
            near_fall_minimum_root_height=config.near_fall_minimum_root_height,
            near_fall_minimum_up_alignment=config.near_fall_minimum_up_alignment,
        )
    )

    rng = jax.random.PRNGKey(config.seed)
    episode = 0
    episode_seed = config.seed
    timestep = 0
    rng, reset_rng = jax.random.split(rng)
    state = reset(reset_rng)
    command_transitions = 0
    previous_command = None
    finite_values = True
    no_opponent_fields = True
    schema_consistent = True
    starts_upright: bool | None = None

    with JsonlDatasetWriter(config.output) as writer:
        while writer.count < config.samples:
            command = jp.asarray(state.info["command"])
            rng, action_rng = jax.random.split(rng)
            control = act(state.obs, action_rng)
            sample = extract(
                state,
                command,
                control.normalized_action,
                control.joint_position_targets,
            )
            record = locomotion_record(
                sample,
                dataset_kind="ordinary_random_commands",
                episode=episode,
                episode_seed=episode_seed,
                timestep=timestep,
                state=state,
            )
            if starts_upright is None:
                starts_upright = not record["fallen"]
            finite_values &= bool(
                np.isfinite(
                    np.asarray(
                        record["root_position_world"]
                        + record["root_linear_velocity_world"]
                        + record["command_velocity"]
                        + record["joint_position"]
                        + record["joint_velocity"]
                    )
                ).all()
            )
            no_opponent_fields &= not any(key.startswith("opponent_") for key in record)
            schema_consistent &= (
                record["schema_version"] == LOCOMOTION_DATASET_SCHEMA_VERSION
            )
            writer.write(record)
            command_array = np.asarray(record["command_velocity"])
            if previous_command is not None and not np.array_equal(
                command_array, previous_command
            ):
                command_transitions += 1
            previous_command = command_array

            state = step(state, control.normalized_action)
            timestep += 1
            if bool(np.asarray(state.done)) and writer.count < config.samples:
                episode += 1
                episode_seed = config.seed + episode
                rng = jax.random.PRNGKey(episode_seed)
                rng, reset_rng = jax.random.split(rng)
                state = reset(reset_rng)
                timestep = 0
                previous_command = None

    checks = {
        "checkpoint_loaded": controller.checkpoint_path == config.checkpoint.resolve(),
        "exact_sample_budget": writer.count == config.samples,
        "finite_values": finite_values,
        "flat_terrain": bool(
            np.asarray(
                environment.mj_model.geom("floor").type == mujoco.mjtGeom.mjGEOM_PLANE
            ).all()
        ),
        "gpu_backend": jax.default_backend() == "gpu",
        "no_opponent_fields": no_opponent_fields,
        "pushes_disabled": not bool(environment_config.push_config.enable),
        "schema_consistent": schema_consistent,
        "starts_upright": bool(starts_upright),
    }
    manifest = {
        "backend": jax.default_backend(),
        "checkpoint": str(controller.checkpoint_path),
        "checks": checks,
        "command_limits": {
            "x": list(environment_config.lin_vel_x),
            "y": list(environment_config.lin_vel_y),
            "yaw": list(environment_config.ang_vel_yaw),
        },
        "command_transitions": command_transitions,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "manifest": str(config.manifest),
            "output": str(config.output),
        },
        "control_timestep": float(environment.dt),
        "dataset_kind": "ordinary_random_commands",
        "episodes": episode + 1,
        "experiment": "p6_random_command_dataset",
        "jax_version": jax.__version__,
        "mujoco_version": version("mujoco"),
        "output_sha256": writer.sha256,
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "samples": writer.count,
        "schema_version": LOCOMOTION_DATASET_SCHEMA_VERSION,
        "seed": config.seed,
    }
    config.manifest.parent.mkdir(parents=True, exist_ok=True)
    config.manifest.write_text(
        json.dumps(manifest, default=json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, default=json_default, sort_keys=True))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
