"""Generate the P6 opponent-free terrain-curriculum locomotion dataset.

The generator advances from flat-only episodes to rough-heightfield episodes
while retaining ordinary structured velocity commands. It writes the shared
single-agent JSONL schema and a manifest with terrain and curriculum coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import mujoco
import numpy as np

from motionforge.cli import run_hydra
from motionforge.controllers import G1LocomotionController
from motionforge.datasets import (
    LOCOMOTION_DATASET_SCHEMA_VERSION,
    TerrainCurriculum,
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
    episode_steps: int = 128
    rough_probabilities: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0)
    naconmax: int = 16
    njmax: int = 128
    fall_minimum_root_height: float = 0.45
    fall_minimum_up_alignment: float = 0.50
    near_fall_minimum_root_height: float = 0.60
    near_fall_minimum_up_alignment: float = 0.80
    output: Path = Path("logs/p6/terrain_curriculum_smoke.jsonl")
    manifest: Path = Path("logs/p6/terrain_curriculum_smoke_manifest.json")


def validate_config(config: Config) -> None:
    if config.samples <= 0:
        raise ValueError("samples must be positive")
    if config.episode_steps <= 0:
        raise ValueError("episode_steps must be positive")
    if config.samples < config.episode_steps * len(config.rough_probabilities):
        raise ValueError("samples must provide at least one episode per stage")
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
    curriculum = TerrainCurriculum(config.rough_probabilities, config.seed)
    environments = {}
    for name, task in (("flat", "flat_terrain"), ("rough", "rough_terrain")):
        environment_config = default_config()
        environment_config.impl = "warp"
        environment_config.naconmax = config.naconmax
        environment_config.njmax = config.njmax
        environment_config.push_config.enable = False
        environments[name] = G1StandingJoystick(
            config=environment_config,
            task=task,
        )

    controller = G1LocomotionController.from_checkpoint(
        config.checkpoint,
        environments["flat"],
        deterministic=True,
    )
    act = jax.jit(controller.act)
    resets = {
        name: jax.jit(environment.reset) for name, environment in environments.items()
    }
    steps = {
        name: jax.jit(environment.step) for name, environment in environments.items()
    }
    extracts = {
        name: jax.jit(
            lambda state, command, action, targets, environment=environment: (
                extract_g1_locomotion_sample(
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
        )
        for name, environment in environments.items()
    }

    episode = 0
    episode_seed = config.seed
    timestep = 0
    stage, rough_probability, terrain = curriculum.assignment(episode, 0.0)
    rng = jax.random.PRNGKey(episode_seed)
    rng, reset_rng = jax.random.split(rng)
    state = resets[terrain](reset_rng)
    records: list[dict] = []
    terrain_counts: Counter[str] = Counter()
    stage_counts: Counter[int] = Counter()
    terminated_episodes = 0

    while len(records) < config.samples:
        command = state.info["command"]
        rng, action_rng = jax.random.split(rng)
        control = act(state.obs, action_rng)
        sample = extracts[terrain](
            state,
            command,
            control.normalized_action,
            control.joint_position_targets,
        )
        record = locomotion_record(
            sample,
            dataset_kind="terrain_curriculum",
            episode=episode,
            episode_seed=episode_seed,
            timestep=timestep,
            state=state,
            extra={
                "curriculum_stage": stage,
                "rough_probability": rough_probability,
                "terrain_kind": terrain,
            },
        )
        records.append(record)
        terrain_counts[terrain] += 1
        stage_counts[stage] += 1

        state = steps[terrain](state, control.normalized_action)
        timestep += 1
        terminated = bool(np.asarray(state.done))
        truncate = timestep >= config.episode_steps
        if (terminated or truncate) and len(records) < config.samples:
            terminated_episodes += int(terminated)
            episode += 1
            episode_seed = config.seed + episode
            progress = len(records) / config.samples
            stage, rough_probability, terrain = curriculum.assignment(
                episode,
                progress,
            )
            rng = jax.random.PRNGKey(episode_seed)
            rng, reset_rng = jax.random.split(rng)
            state = resets[terrain](reset_rng)
            timestep = 0

    serialized = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(serialized, encoding="utf-8")

    commands = np.asarray([record["command_velocity"] for record in records])
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
        "all_curriculum_stages_present": set(stage_counts)
        == set(range(len(config.rough_probabilities))),
        "both_terrains_present": set(terrain_counts) == {"flat", "rough"},
        "checkpoint_loaded": controller.checkpoint_path == config.checkpoint.resolve(),
        "commands_in_controller_range": bool(
            np.all(commands[:, 0] >= environments["flat"]._config.lin_vel_x[0])
            and np.all(commands[:, 0] <= environments["flat"]._config.lin_vel_x[1])
            and np.all(commands[:, 1] >= environments["flat"]._config.lin_vel_y[0])
            and np.all(commands[:, 1] <= environments["flat"]._config.lin_vel_y[1])
            and np.all(commands[:, 2] >= environments["flat"]._config.ang_vel_yaw[0])
            and np.all(commands[:, 2] <= environments["flat"]._config.ang_vel_yaw[1])
        ),
        "exact_sample_budget": len(records) == config.samples,
        "finite_values": bool(numeric_finite),
        "gpu_backend": jax.default_backend() == "gpu",
        "no_opponent_fields": all(
            not any(key.startswith("opponent_") for key in record) for record in records
        ),
        "pushes_disabled": all(
            not bool(environment._config.push_config.enable)
            for environment in environments.values()
        ),
        "schema_consistent": all(
            record["schema_version"] == LOCOMOTION_DATASET_SCHEMA_VERSION
            for record in records
        ),
        "starts_upright": not records[0]["fallen"],
        "terrain_models": bool(
            np.asarray(
                environments["flat"].mj_model.geom("floor").type
                == mujoco.mjtGeom.mjGEOM_PLANE
            ).all()
            and np.asarray(
                environments["rough"].mj_model.geom("floor").type
                == mujoco.mjtGeom.mjGEOM_HFIELD
            ).all()
        ),
    }
    manifest = {
        "backend": jax.default_backend(),
        "checkpoint": str(controller.checkpoint_path),
        "checks": checks,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "manifest": str(config.manifest),
            "output": str(config.output),
        },
        "control_timestep": float(environments["flat"].dt),
        "dataset_kind": "terrain_curriculum",
        "episodes": episode + 1,
        "experiment": "p6_terrain_curriculum_dataset",
        "jax_version": jax.__version__,
        "mujoco_version": version("mujoco"),
        "output_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "passed": bool(all(checks.values())),
        "planned_episodes": math.ceil(config.samples / config.episode_steps),
        "python_version": platform.python_version(),
        "samples": len(records),
        "schema_version": LOCOMOTION_DATASET_SCHEMA_VERSION,
        "seed": config.seed,
        "stage_counts": dict(stage_counts),
        "terrain_counts": dict(terrain_counts),
        "terminated_episodes": terminated_episodes,
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
