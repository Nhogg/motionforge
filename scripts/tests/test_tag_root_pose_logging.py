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
    build_tag_flat_terrain_layout,
    build_tag_foot_contact_layout,
    build_tag_joint_state_layout,
    build_tag_root_pose_layout,
    build_tag_root_velocity_layout,
    build_tag_tracking_layout,
    derive_tag_linear_acceleration,
    derive_tag_yaw_acceleration,
    extract_tag_flat_terrain,
    extract_tag_foot_contact,
    extract_tag_joint_state,
    extract_tag_root_pose,
    extract_tag_root_velocity,
    extract_tag_tracking_error,
    tag_controller_command,
    tag_opponent_relative_state,
    tag_reward_outcome,
    tag_stability_state,
)


@dataclass
class Config:
    seed: int = 0
    steps: int = 10
    separation: float = 2.0
    naconmax: int = 32
    njmax: int = 256
    near_fall_minimum_root_height: float = 0.60
    near_fall_minimum_up_alignment: float = 0.80
    output: Path = Path("logs/p5/tag_yaw_acceleration_l.jsonl")
    summary: Path = Path("logs/p5/tag_yaw_acceleration_l_summary.json")


def _validate_config(config: Config) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.separation <= 0.0:
        raise ValueError("separation must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("contact capacities must be positive")
    if config.near_fall_minimum_root_height <= 0.0:
        raise ValueError("near_fall_minimum_root_height must be positive")
    if not 0.0 <= config.near_fall_minimum_up_alignment <= 1.0:
        raise ValueError("near_fall_minimum_up_alignment must be in [0, 1]")


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
    terrain_layout = build_tag_flat_terrain_layout(environment.model_bundle)
    tracking_layout = build_tag_tracking_layout(environment.model_bundle)
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
    reward = jp.zeros((2,))
    pursuer_index = next(
        agent.index for agent in environment.roles.agents if agent.role == "pursuer"
    )
    records = []
    expected_tracking_errors = []
    for timestep in range(config.steps + 1):
        root_pose, root_velocity, joint_state, foot_contact = extract(state)
        controller_command = tag_controller_command(
            velocity_command,
            joint_position_target,
        )
        relative_state = tag_opponent_relative_state(
            state.observation.relative_position,
            state.observation.relative_velocity,
        )
        terrain_state = extract_tag_flat_terrain(terrain_layout)
        reward_outcome = tag_reward_outcome(
            reward,
            state.termination.tagged,
            state.termination.fallen,
            state.termination.out_of_bounds,
            state.termination.timed_out,
            pursuer_index,
        )
        tracking_error = extract_tag_tracking_error(
            state.data,
            tracking_layout,
            velocity_command,
        )
        stability_state = tag_stability_state(
            state.diagnostics.fall_detected,
            state.diagnostics.root_height,
            state.diagnostics.up_alignment,
            config.near_fall_minimum_root_height,
            config.near_fall_minimum_up_alignment,
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
        opponent_relative_position = np.asarray(relative_state.position)
        opponent_relative_velocity = np.asarray(relative_state.velocity)
        terrain_height = np.asarray(terrain_state.height)
        terrain_normal_world = np.asarray(terrain_state.normal_world)
        game_reward = np.asarray(reward_outcome.reward)
        command_tracking_error = np.asarray(tracking_error.velocity)
        stability_fallen = np.asarray(stability_state.fallen)
        stability_near_fall = np.asarray(stability_state.near_fall)
        stability_root_height = np.asarray(stability_state.root_height)
        stability_up_alignment = np.asarray(stability_state.up_alignment)
        measured_controller_velocity = np.stack(
            [
                np.asarray(
                    [
                        state.data.sensordata[linear_slice][0],
                        state.data.sensordata[linear_slice][1],
                        state.data.sensordata[angular_slice][2],
                    ]
                )
                for linear_slice, angular_slice in zip(
                    tracking_layout.pelvis_linear_velocity_slices,
                    tracking_layout.pelvis_angular_velocity_slices,
                    strict=True,
                )
            ]
        )
        expected_tracking_errors.append(command_velocity - measured_controller_velocity)
        records.append(
            {
                "command_joint_position_target": (
                    command_joint_position_target.tolist()
                ),
                "command_velocity": command_velocity.tolist(),
                "command_velocity_tracking_error": command_tracking_error.tolist(),
                "episode_seed": config.seed,
                "foot_contact": foot_contact_active.tolist(),
                "foot_contact_normal_force": foot_contact_normal_force.tolist(),
                "game_done": bool(np.asarray(reward_outcome.done)),
                "game_fallen": np.asarray(reward_outcome.fallen).tolist(),
                "game_out_of_bounds": np.asarray(reward_outcome.out_of_bounds).tolist(),
                "game_reward": game_reward.tolist(),
                "game_tagged": bool(np.asarray(reward_outcome.tagged)),
                "game_timed_out": bool(np.asarray(reward_outcome.timed_out)),
                "game_winner_index": int(np.asarray(reward_outcome.winner_index)),
                "joint_position": joint_position.tolist(),
                "joint_velocity": joint_velocity.tolist(),
                "orientation_wxyz": orientation.tolist(),
                "opponent_relative_position_heading": (
                    opponent_relative_position.tolist()
                ),
                "opponent_relative_velocity_heading": (
                    opponent_relative_velocity.tolist()
                ),
                "pelvis_angular_velocity": pelvis_angular_velocity.tolist(),
                "position": position.tolist(),
                "schema_version": TAG_TRAJECTORY_SCHEMA_VERSION,
                "simulation_time": float(np.asarray(state.diagnostics.elapsed_seconds)),
                "stability_fallen": stability_fallen.tolist(),
                "stability_near_fall": stability_near_fall.tolist(),
                "stability_root_height": stability_root_height.tolist(),
                "stability_up_alignment": stability_up_alignment.tolist(),
                "terrain_height": terrain_height.tolist(),
                "terrain_kind": "flat_plane",
                "terrain_normal_world": terrain_normal_world.tolist(),
                "timestep": timestep,
                "world_linear_velocity": world_linear_velocity.tolist(),
            }
        )
        if timestep < config.steps:
            state = step(state, joint_position_target)

    positions = np.asarray([record["position"] for record in records])
    orientations = np.asarray([record["orientation_wxyz"] for record in records])
    world_linear_velocities = np.asarray(
        [record["world_linear_velocity"] for record in records]
    )
    linear_acceleration = derive_tag_linear_acceleration(
        jp.asarray(world_linear_velocities),
        environment.config.control_timestep,
    )
    world_linear_accelerations = np.asarray(linear_acceleration.world)
    linear_acceleration_valid = np.asarray(linear_acceleration.valid)
    for index, record in enumerate(records):
        record["world_linear_acceleration"] = world_linear_accelerations[index].tolist()
        record["world_linear_acceleration_valid"] = bool(
            linear_acceleration_valid[index]
        )

    pelvis_angular_velocities = np.asarray(
        [record["pelvis_angular_velocity"] for record in records]
    )
    yaw_acceleration = derive_tag_yaw_acceleration(
        jp.asarray(pelvis_angular_velocities),
        environment.config.control_timestep,
    )
    pelvis_yaw_accelerations = np.asarray(yaw_acceleration.pelvis)
    yaw_acceleration_valid = np.asarray(yaw_acceleration.valid)
    for index, record in enumerate(records):
        record["pelvis_yaw_acceleration"] = pelvis_yaw_accelerations[index].tolist()
        record["pelvis_yaw_acceleration_valid"] = bool(yaw_acceleration_valid[index])

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
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
    opponent_relative_positions = np.asarray(
        [record["opponent_relative_position_heading"] for record in records]
    )
    opponent_relative_velocities = np.asarray(
        [record["opponent_relative_velocity_heading"] for record in records]
    )
    terrain_heights = np.asarray([record["terrain_height"] for record in records])
    terrain_normals = np.asarray([record["terrain_normal_world"] for record in records])
    game_rewards = np.asarray([record["game_reward"] for record in records])
    command_tracking_errors = np.asarray(
        [record["command_velocity_tracking_error"] for record in records]
    )
    expected_tracking_errors = np.asarray(expected_tracking_errors)
    stability_falls = np.asarray([record["stability_fallen"] for record in records])
    stability_near_falls = np.asarray(
        [record["stability_near_fall"] for record in records]
    )
    stability_root_heights = np.asarray(
        [record["stability_root_height"] for record in records]
    )
    stability_up_alignments = np.asarray(
        [record["stability_up_alignment"] for record in records]
    )
    quaternion_norms = np.linalg.norm(orientations, axis=-1)
    checks = {
        "backend_gpu": jax.default_backend() == "gpu",
        "finite_values": bool(
            np.isfinite(positions).all()
            and np.isfinite(orientations).all()
            and np.isfinite(world_linear_velocities).all()
            and np.isfinite(pelvis_angular_velocities).all()
            and np.isfinite(pelvis_yaw_accelerations).all()
            and np.isfinite(joint_positions).all()
            and np.isfinite(joint_velocities).all()
            and np.isfinite(command_velocities).all()
            and np.isfinite(command_joint_position_targets).all()
            and np.isfinite(foot_contact_normal_forces).all()
            and np.isfinite(opponent_relative_positions).all()
            and np.isfinite(opponent_relative_velocities).all()
            and np.isfinite(terrain_heights).all()
            and np.isfinite(terrain_normals).all()
            and np.isfinite(game_rewards).all()
            and np.isfinite(command_tracking_errors).all()
            and np.isfinite(stability_root_heights).all()
            and np.isfinite(stability_up_alignments).all()
            and np.isfinite(world_linear_accelerations).all()
        ),
        "foot_contact_force_nonnegative": bool(
            np.all(foot_contact_normal_forces >= -1e-5)
        ),
        "foot_contact_force_shape": foot_contact_normal_forces.shape
        == (config.steps + 1, 2, 2),
        "foot_contact_observed": bool(np.any(foot_contacts)),
        "foot_contact_shape": foot_contacts.shape == (config.steps + 1, 2, 2),
        "game_outcome_ongoing": all(
            not record["game_done"] and record["game_winner_index"] == -1
            for record in records
        ),
        "game_reward_shape": game_rewards.shape == (config.steps + 1, 2),
        "initial_relative_distance": bool(
            np.allclose(
                np.linalg.norm(opponent_relative_positions[0], axis=1),
                config.separation,
                atol=1e-5,
            )
        ),
        "command_joint_target_shape": command_joint_position_targets.shape
        == (config.steps + 1, 2, 29),
        "command_velocity_shape": command_velocities.shape == (config.steps + 1, 2, 3),
        "command_tracking_error_shape": command_tracking_errors.shape
        == (config.steps + 1, 2, 3),
        "command_tracking_error_values": bool(
            np.allclose(command_tracking_errors, expected_tracking_errors)
        ),
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
        "linear_acceleration_shape": world_linear_accelerations.shape
        == (config.steps + 1, 2, 3),
        "linear_acceleration_validity": bool(
            not linear_acceleration_valid[0] and np.all(linear_acceleration_valid[1:])
        ),
        "orientation_shape": orientations.shape == (config.steps + 1, 2, 4),
        "yaw_acceleration_shape": pelvis_yaw_accelerations.shape
        == (config.steps + 1, 2),
        "yaw_acceleration_validity": bool(
            not yaw_acceleration_valid[0] and np.all(yaw_acceleration_valid[1:])
        ),
        "relative_position_shape": opponent_relative_positions.shape
        == (config.steps + 1, 2, 2),
        "relative_velocity_shape": opponent_relative_velocities.shape
        == (config.steps + 1, 2, 2),
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
        "stability_exclusive": bool(not np.any(stability_falls & stability_near_falls)),
        "stability_shape": (
            stability_falls.shape
            == stability_near_falls.shape
            == stability_root_heights.shape
            == stability_up_alignments.shape
            == (config.steps + 1, 2)
        ),
        "terrain_height_matches_floor": bool(
            np.allclose(terrain_heights, terrain_layout.height)
        ),
        "terrain_height_shape": terrain_heights.shape == (config.steps + 1, 2),
        "terrain_kind": all(
            record["terrain_kind"] == "flat_plane" for record in records
        ),
        "terrain_normal_shape": terrain_normals.shape == (config.steps + 1, 2, 3),
        "terrain_normals_unit": bool(
            np.allclose(np.linalg.norm(terrain_normals, axis=-1), 1.0, atol=1e-7)
        ),
        "timesteps_contiguous": [record["timestep"] for record in records]
        == list(range(config.steps + 1)),
    }
    tag_fixture = tag_reward_outcome(
        jp.asarray([1.0, -1.0]),
        jp.asarray(True),
        jp.zeros((2,), dtype=bool),
        jp.zeros((2,), dtype=bool),
        jp.asarray(False),
        pursuer_index,
    )
    timeout_fixture = tag_reward_outcome(
        jp.zeros((2,)),
        jp.asarray(False),
        jp.zeros((2,), dtype=bool),
        jp.zeros((2,), dtype=bool),
        jp.asarray(True),
        pursuer_index,
    )
    checks["tag_winner_is_pursuer"] = (
        int(np.asarray(tag_fixture.winner_index)) == pursuer_index
    )
    checks["timeout_winner_is_evader"] = (
        int(np.asarray(timeout_fixture.winner_index)) == 1 - pursuer_index
    )
    stability_fixture = tag_stability_state(
        jp.asarray([False, True]),
        jp.asarray([config.near_fall_minimum_root_height - 0.01, 0.1]),
        jp.asarray([1.0, 0.0]),
        config.near_fall_minimum_root_height,
        config.near_fall_minimum_up_alignment,
    )
    checks["near_fall_fixture"] = bool(
        np.array_equal(np.asarray(stability_fixture.near_fall), [True, False])
    )
    acceleration_fixture = derive_tag_linear_acceleration(
        jp.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.2, 0.0, 0.0], [1.0, -0.4, 0.0]],
            ]
        ),
        0.2,
    )
    checks["linear_acceleration_fixture"] = bool(
        np.allclose(
            np.asarray(acceleration_fixture.world[1]),
            [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]],
        )
    )
    yaw_acceleration_fixture = derive_tag_yaw_acceleration(
        jp.asarray(
            [
                [[0.0, 0.0, 0.2], [0.0, 0.0, -0.4]],
                [[0.0, 0.0, 0.5], [0.0, 0.0, -0.2]],
            ]
        ),
        0.1,
    )
    checks["yaw_acceleration_fixture"] = bool(
        np.allclose(np.asarray(yaw_acceleration_fixture.pelvis[1]), [3.0, 2.0])
    )
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
            "summary": str(config.summary),
        },
        "experiment": "tag_yaw_acceleration_logging",
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
