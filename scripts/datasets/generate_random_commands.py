"""Generate the P6 ordinary random-command locomotion dataset.

Inputs are a deterministic locomotion checkpoint, an initial seed, and an exact
agent-control-step budget. Outputs are JSONL timestep records and a JSON
manifest. The generator uses flat terrain, disables pushes, and leaves the
canonical structured command sampler unchanged.
"""

from __future__ import annotations

import hashlib
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
    extract_g1_locomotion_sample,
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


def _record(sample, *, episode: int, episode_seed: int, timestep: int, state) -> dict:
    host = jax.tree.map(lambda value: np.asarray(value), sample)
    return {
        "command_tracking_error": host.command_tracking_error.tolist(),
        "command_velocity": host.command_velocity.tolist(),
        "dataset_kind": "ordinary_random_commands",
        "done": bool(np.asarray(state.done)),
        "episode": episode,
        "episode_seed": episode_seed,
        "fallen": bool(host.fallen),
        "foot_contact": host.foot_contact.tolist(),
        "foot_position_world": host.foot_position_world.tolist(),
        "joint_position": host.joint_position.tolist(),
        "joint_position_target": host.joint_position_target.tolist(),
        "joint_velocity": host.joint_velocity.tolist(),
        "near_fall": bool(host.near_fall),
        "normalized_action": host.normalized_action.tolist(),
        "pelvis_angular_velocity": host.pelvis_angular_velocity.tolist(),
        "reward": float(np.asarray(state.reward)),
        "root_height": float(host.root_height),
        "root_linear_velocity_local": host.root_linear_velocity_local.tolist(),
        "root_linear_velocity_world": host.root_linear_velocity_world.tolist(),
        "root_orientation_wxyz": host.root_orientation_wxyz.tolist(),
        "root_position_world": host.root_position_world.tolist(),
        "schema_version": LOCOMOTION_DATASET_SCHEMA_VERSION,
        "timestep": timestep,
        "up_alignment": float(host.up_alignment),
    }


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
    records: list[dict] = []
    command_transitions = 0
    previous_command = None

    while len(records) < config.samples:
        command = jp.asarray(state.info["command"])
        rng, action_rng = jax.random.split(rng)
        control = act(state.obs, action_rng)
        sample = extract(
            state,
            command,
            control.normalized_action,
            control.joint_position_targets,
        )
        record = _record(
            sample,
            episode=episode,
            episode_seed=episode_seed,
            timestep=timestep,
            state=state,
        )
        records.append(record)
        command_array = np.asarray(record["command_velocity"])
        if previous_command is not None and not np.array_equal(
            command_array, previous_command
        ):
            command_transitions += 1
        previous_command = command_array

        state = step(state, control.normalized_action)
        timestep += 1
        if bool(np.asarray(state.done)) and len(records) < config.samples:
            episode += 1
            episode_seed = config.seed + episode
            rng = jax.random.PRNGKey(episode_seed)
            rng, reset_rng = jax.random.split(rng)
            state = reset(reset_rng)
            timestep = 0
            previous_command = None

    serialized = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(serialized, encoding="utf-8")

    numeric_finite = all(
        np.isfinite(
            np.asarray(
                record["root_position_world"]
                + record["root_linear_velocity_world"]
                + record["command_velocity"]
                + record["joint_position"]
                + record["joint_velocity"]
            )
        ).all()
        for record in records
    )
    checks = {
        "checkpoint_loaded": controller.checkpoint_path == config.checkpoint.resolve(),
        "exact_sample_budget": len(records) == config.samples,
        "finite_values": bool(numeric_finite),
        "flat_terrain": bool(
            np.asarray(
                environment.mj_model.geom("floor").type == mujoco.mjtGeom.mjGEOM_PLANE
            ).all()
        ),
        "gpu_backend": jax.default_backend() == "gpu",
        "no_opponent_fields": all(
            not any(key.startswith("opponent_") for key in record) for record in records
        ),
        "pushes_disabled": not bool(environment_config.push_config.enable),
        "schema_consistent": all(
            record["schema_version"] == LOCOMOTION_DATASET_SCHEMA_VERSION
            for record in records
        ),
        "starts_upright": not records[0]["fallen"],
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
        "output_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "samples": len(records),
        "schema_version": LOCOMOTION_DATASET_SCHEMA_VERSION,
        "seed": config.seed,
    }
    config.manifest.parent.mkdir(parents=True, exist_ok=True)
    config.manifest.write_text(
        json.dumps(manifest, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, default=_json_default, sort_keys=True))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
