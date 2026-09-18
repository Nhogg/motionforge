"""Record and validate per-timestep root poses from the tag environment.

The seeded smoke experiment writes append-only JSONL trajectory records and a
JSON summary. It uses default joint targets so it tests logging independently
of a learned controller.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import TagEnvironmentConfig, TwoG1TagEnvironment
from motionforge.logging import (
    TAG_TRAJECTORY_SCHEMA_VERSION,
    build_tag_foot_contact_layout,
    build_tag_joint_state_layout,
    build_tag_root_pose_layout,
    build_tag_root_velocity_layout,
    extract_tag_foot_contact,
    extract_tag_joint_state,
    extract_tag_root_pose,
    extract_tag_root_velocity,
    tag_controller_command,
)


@dataclass
class Config:
    seed: int = 0
    steps: int = 10
    separation: float = 2.0
    naconmax: int = 32
    njmax: int = 256
    output: Path = Path("logs/p5/tag_contact_state_e.jsonl")
    summary: Path = Path("logs/p5/tag_contact_state_e_summary.json")


def _validate_config(config: Config) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.separation <= 0.0:
        raise ValueError("separation must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("contact capacities must be positive")


def main(config: Config) -> None:
    _validate_config(config)
    environment = TwoG1TagEnvironment(
        TagEnvironmentConfig(
            separation=config.separation,
            naconmax=config.naconmax,
            njmax=config.njmax,
        )
    )
    pose_layout = build_tag_root_pose_layout(environment.model_bundle)
    velocity_layout = build_tag_root_velocity_layout(environment.model_bundle)
    joint_layout = build_tag_joint_state_layout(environment.model_bundle)
    foot_contact_layout = build_tag_foot_contact_layout(environment.model_bundle)
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    extract = jax.jit(
        lambda state: (
            extract_tag_root_pose(state.data, pose_layout),
            extract_tag_root_velocity(state.data, velocity_layout),
            extract_tag_joint_state(state.data, joint_layout),
            extract_tag_foot_contact(state.data, foot_contact_layout),
        )
    )

    state = reset(jax.random.PRNGKey(config.seed))
    velocity_command = jp.zeros((2, 3))
    joint_position_target = environment.default_joint_targets
    records = []
    for timestep in range(config.steps + 1):
        root_pose, root_velocity, joint_state, foot_contact = extract(state)
        controller_command = tag_controller_command(
            velocity_command,
            joint_position_target,
        )
        root_pose.position.block_until_ready()
        position = np.asarray(root_pose.position)
        orientation = np.asarray(root_pose.orientation_wxyz)
        world_linear_velocity = np.asarray(root_velocity.world_linear)
        pelvis_angular_velocity = np.asarray(root_velocity.pelvis_angular)
        joint_position = np.asarray(joint_state.position)
        joint_velocity = np.asarray(joint_state.velocity)
        command_velocity = np.asarray(controller_command.velocity)
        command_joint_position_target = np.asarray(
            controller_command.joint_position_target
        )
        foot_contact_active = np.asarray(foot_contact.active)
        foot_contact_normal_force = np.asarray(foot_contact.normal_force)
        records.append(
            {
                "command_joint_position_target": (
                    command_joint_position_target.tolist()
                ),
                "command_velocity": command_velocity.tolist(),
                "episode_seed": config.seed,
                "foot_contact": foot_contact_active.tolist(),
                "foot_contact_normal_force": foot_contact_normal_force.tolist(),
                "joint_position": joint_position.tolist(),
                "joint_velocity": joint_velocity.tolist(),
                "orientation_wxyz": orientation.tolist(),
                "pelvis_angular_velocity": pelvis_angular_velocity.tolist(),
                "position": position.tolist(),
                "schema_version": TAG_TRAJECTORY_SCHEMA_VERSION,
                "simulation_time": float(np.asarray(state.diagnostics.elapsed_seconds)),
                "timestep": timestep,
                "world_linear_velocity": world_linear_velocity.tolist(),
            }
        )
        if timestep < config.steps:
            state = step(state, joint_position_target)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    positions = np.asarray([record["position"] for record in records])
    orientations = np.asarray([record["orientation_wxyz"] for record in records])
    world_linear_velocities = np.asarray(
        [record["world_linear_velocity"] for record in records]
    )
    pelvis_angular_velocities = np.asarray(
        [record["pelvis_angular_velocity"] for record in records]
    )
    joint_positions = np.asarray([record["joint_position"] for record in records])
    joint_velocities = np.asarray([record["joint_velocity"] for record in records])
    command_velocities = np.asarray([record["command_velocity"] for record in records])
    command_joint_position_targets = np.asarray(
        [record["command_joint_position_target"] for record in records]
    )
    foot_contacts = np.asarray([record["foot_contact"] for record in records])
    foot_contact_normal_forces = np.asarray(
        [record["foot_contact_normal_force"] for record in records]
    )
    quaternion_norms = np.linalg.norm(orientations, axis=-1)
    checks = {
        "backend_gpu": jax.default_backend() == "gpu",
        "finite_values": bool(
            np.isfinite(positions).all()
            and np.isfinite(orientations).all()
            and np.isfinite(world_linear_velocities).all()
            and np.isfinite(pelvis_angular_velocities).all()
            and np.isfinite(joint_positions).all()
            and np.isfinite(joint_velocities).all()
            and np.isfinite(command_velocities).all()
            and np.isfinite(command_joint_position_targets).all()
            and np.isfinite(foot_contact_normal_forces).all()
        ),
        "foot_contact_force_nonnegative": bool(
            np.all(foot_contact_normal_forces >= -1e-5)
        ),
        "foot_contact_force_shape": foot_contact_normal_forces.shape
        == (config.steps + 1, 2, 2),
        "foot_contact_observed": bool(np.any(foot_contacts)),
        "foot_contact_shape": foot_contacts.shape == (config.steps + 1, 2, 2),
        "command_joint_target_shape": command_joint_position_targets.shape
        == (config.steps + 1, 2, 29),
        "command_velocity_shape": command_velocities.shape == (config.steps + 1, 2, 3),
        "commands_match_applied_controls": bool(
            np.allclose(
                command_joint_position_targets[-1],
                np.asarray(state.data.ctrl)[np.asarray(environment.actuator_ids)],
            )
        ),
        "joint_position_shape": joint_positions.shape == (config.steps + 1, 2, 29),
        "joint_velocity_shape": joint_velocities.shape == (config.steps + 1, 2, 29),
        "angular_velocity_shape": pelvis_angular_velocities.shape
        == (config.steps + 1, 2, 3),
        "linear_velocity_shape": world_linear_velocities.shape
        == (config.steps + 1, 2, 3),
        "orientation_shape": orientations.shape == (config.steps + 1, 2, 4),
        "positions_distinct": bool(
            np.linalg.norm(positions[0, 1] - positions[0, 0]) > 0.0
        ),
        "position_shape": positions.shape == (config.steps + 1, 2, 3),
        "quaternions_normalized": bool(np.allclose(quaternion_norms, 1.0, atol=1e-5)),
        "record_count": len(records) == config.steps + 1,
        "simulation_time": bool(
            np.isclose(
                records[-1]["simulation_time"],
                config.steps * environment.config.control_timestep,
            )
        ),
        "timesteps_contiguous": [record["timestep"] for record in records]
        == list(range(config.steps + 1)),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
            "summary": str(config.summary),
        },
        "experiment": "tag_contact_state_logging",
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "record_count": len(records),
        "schema_version": TAG_TRAJECTORY_SCHEMA_VERSION,
        "seed": config.seed,
    }
    config.summary.parent.mkdir(parents=True, exist_ok=True)
    config.summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
