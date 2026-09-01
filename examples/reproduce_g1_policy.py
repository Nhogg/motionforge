from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort
import tyro
from mujoco_playground._src.locomotion.g1 import g1_constants
from mujoco_playground._src.locomotion.g1.base import get_assets

PLAYGROUND_REVISION = "8a4b4642d8eba8a80ac99ed125cb62c16e1457ad"


@dataclass(frozen=True)
class Config:
    """Reproduce the bundled MuJoCo Playground G1 policy."""

    seed: int = 0

    duration: float = 10.0

    command_x: float = 0.5

    command_y: float = 0.0

    command_yaw: float = 0.0

    stop_at: float | None = None

    freeze_phase_at_zero: bool = False

    hold_pose_at_zero: bool = False

    output: Path | None = None


class FixedCommandController:
    """Run the bundled ONNX policy using a constant velocity command."""

    def __init__(
        self,
        policy_path: Path,
        default_angles: np.ndarray,
        command: np.ndarray,
        control_timestep: float,
        simulation_timestep: float,
        action_scale: float = 0.5,
        freeze_phase_at_zero: bool = False,
        hold_pose_at_zero: bool = False,
    ) -> None:
        self._policy = ort.InferenceSession(
            policy_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
        self._output_names = ["continuous_actions"]

        self._default_angles = default_angles
        self._command = command.astype(np.float32)
        self._action_scale = action_scale
        self._freeze_phase_at_zero = freeze_phase_at_zero
        self._hold_pose_at_zero = hold_pose_at_zero

        self._last_action = np.zeros_like(
            default_angles,
            dtype=np.float32,
        )

        self._phase = np.array([0.0, np.pi])
        self._gait_frequency = 1.5
        self._phase_delta = 2.0 * np.pi * self._gait_frequency * control_timestep

        self._substeps = int(round(control_timestep / simulation_timestep))
        self._counter = 0
        self.control_updates = 0

    def set_command(self, command: np.ndarray) -> None:
        self._command = command.astype(np.float32)

    def observation(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> np.ndarray:
        linear_velocity = data.sensor("local_linvel_pelvis").data
        angular_velocity = data.sensor("gyro_pelvis").data

        imu_rotation = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)

        gravity = imu_rotation.T @ np.array([0.0, 0.0, -1.0])

        joint_angles = data.qpos[7:] - self._default_angles
        joint_velocities = data.qvel[6:]

        phase = np.concatenate(
            [
                np.cos(self._phase),
                np.sin(self._phase),
            ]
        )

        observation = np.hstack(
            [
                linear_velocity,
                angular_velocity,
                gravity,
                self._command,
                joint_angles,
                joint_velocities,
                self._last_action,
                phase,
            ]
        )

        return observation.astype(np.float32)

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self._counter += 1

        if self._counter % self._substeps != 0:
            return

        if self._hold_pose_at_zero and np.linalg.norm(self._command) < 0.01:
            self._last_action.fill(0.0)
            data.ctrl[:] = self._default_angles
            self.control_updates += 1
            return

        observation = self.observation(model, data)
        policy_input = {"obs": observation.reshape(1, -1)}

        prediction = self._policy.run(
            self._output_names,
            policy_input,
        )[0][0]

        self._last_action = prediction.copy()

        data.ctrl[:] = prediction * self._action_scale + self._default_angles

        if self._freeze_phase_at_zero and np.linalg.norm(self._command) < 0.01:
            self._phase = np.full(2, np.pi)
        else:
            self._phase = (
                np.fmod(
                    self._phase + self._phase_delta + np.pi,
                    2.0 * np.pi,
                )
                - np.pi
            )

        self.control_updates += 1


def array_digest(array: np.ndarray) -> str:
    canonical = np.asarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def root_yaw(data: mujoco.MjData) -> float:
    """Return root yaw in radians from MuJoCo's wxyz free-joint quaternion."""
    w, x, y, z = data.qpos[3:7]
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def run_reproduction(config: Config) -> dict[str, Any]:
    if config.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")
    if config.stop_at is not None and not 0.0 < config.stop_at < config.duration:
        raise ValueError("--stop-at must be between zero and --duration")

    simulation_timestep = 0.002
    control_timestep = 0.02

    policy_path = (
        Path(__file__).resolve().parents[2]
        / "mujoco_playground"
        / "mujoco_playground"
        / "experimental"
        / "sim2sim"
        / "onnx"
        / "g1_policy.onnx"
    )

    if not policy_path.is_file():
        raise FileNotFoundError(f"Bundled policy not found: {policy_path}")

    model = mujoco.MjModel.from_xml_path(
        g1_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        assets=get_assets(),
    )
    data = mujoco.MjData(model)

    keyframe_id = model.keyframe("knees_bent").id
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    model.opt.timestep = simulation_timestep

    default_angles = np.array(model.keyframe("knees_bent").qpos[7:])

    # Hold the initial pose until the first policy update.
    data.ctrl[:] = default_angles
    mujoco.mj_forward(model, data)

    command = np.array(
        [
            config.command_x,
            config.command_y,
            config.command_yaw,
        ],
        dtype=np.float32,
    )

    controller = FixedCommandController(
        policy_path=policy_path,
        default_angles=default_angles,
        command=command,
        control_timestep=control_timestep,
        simulation_timestep=simulation_timestep,
        freeze_phase_at_zero=config.freeze_phase_at_zero,
        hold_pose_at_zero=config.hold_pose_at_zero,
    )

    initial_position = data.qpos[:3].copy()
    initial_yaw = root_yaw(data)
    previous_yaw = initial_yaw
    accumulated_yaw = 0.0
    minimum_root_height = float(data.qpos[2])
    maximum_contacts = int(data.ncon)
    state_is_finite = True
    local_velocity_samples: list[np.ndarray] = []
    yaw_rate_samples: list[float] = []
    command_samples: list[np.ndarray] = []
    active_command = command.copy()
    horizontal_path_length = 0.0
    previous_position = initial_position.copy()
    stop_applied_at: float | None = None
    stop_position: np.ndarray | None = None
    post_stop_path_length = 0.0

    simulation_steps = int(round(config.duration / simulation_timestep))

    for _ in range(simulation_steps):
        if (
            config.stop_at is not None
            and stop_applied_at is None
            and data.time >= config.stop_at
        ):
            active_command = np.zeros(3, dtype=np.float32)
            controller.set_command(active_command)
            stop_applied_at = float(data.time)
            stop_position = data.qpos[:3].copy()

        controller.update(model, data)
        mujoco.mj_step(model, data)

        current_yaw = root_yaw(data)
        yaw_delta = (current_yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
        accumulated_yaw += yaw_delta
        previous_yaw = current_yaw

        current_position = data.qpos[:3].copy()
        horizontal_path_length += float(
            np.linalg.norm(current_position[:2] - previous_position[:2])
        )
        if stop_applied_at is not None:
            post_stop_path_length += float(
                np.linalg.norm(current_position[:2] - previous_position[:2])
            )
        previous_position = current_position

        local_velocity_samples.append(
            data.sensor("local_linvel_pelvis").data[:2].copy()
        )
        yaw_rate_samples.append(float(data.sensor("gyro_pelvis").data[2]))
        command_samples.append(active_command.copy())

        minimum_root_height = min(
            minimum_root_height,
            float(data.qpos[2]),
        )
        maximum_contacts = max(
            maximum_contacts,
            int(data.ncon),
        )

        state_is_finite &= bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.ctrl).all()
        )

    final_position = data.qpos[:3].copy()
    final_yaw = root_yaw(data)
    displacement = final_position - initial_position

    local_velocities = np.asarray(local_velocity_samples)
    yaw_rates = np.asarray(yaw_rate_samples)
    commands = np.asarray(command_samples)
    quarter_steps = max(1, simulation_steps // 4)
    linear_velocity_errors = local_velocities - commands[:, :2]
    yaw_rate_errors = yaw_rates - commands[:, 2]
    final_local_linear_velocity = local_velocities[-1]
    final_window_steps = min(
        simulation_steps,
        max(1, int(round(1.0 / simulation_timestep))),
    )
    final_window_local_velocities = local_velocities[-final_window_steps:]

    if abs(config.command_yaw) > 1.0e-6:
        command_progress = accumulated_yaw * config.command_yaw > 0.1
    elif abs(config.command_x) > 1.0e-6:
        command_progress = float(displacement[0]) * config.command_x > 0.05
    elif abs(config.command_y) > 1.0e-6:
        command_progress = float(displacement[1]) * config.command_y > 0.05
    else:
        command_progress = True

    checks = {
        "controller_updated": controller.control_updates > 0,
        "command_progress": command_progress,
        "no_fall": minimum_root_height > 0.35,
        "state_finite": state_is_finite,
        "simulation_time": math.isclose(
            float(data.time),
            config.duration,
            rel_tol=0.0,
            abs_tol=simulation_timestep,
        ),
    }

    return {
        "checks": checks,
        "command": command.tolist(),
        "control_timestep": control_timestep,
        "control_updates": controller.control_updates,
        "displacement": displacement.tolist(),
        "duration": config.duration,
        "experiment": "mujoco_playground_g1_onnx_policy",
        "final_position": final_position.tolist(),
        "final_yaw": final_yaw,
        "final_local_linear_velocity": final_local_linear_velocity.tolist(),
        "final_local_linear_speed": float(
            np.linalg.norm(final_local_linear_velocity)
        ),
        "final_window_seconds": final_window_steps * simulation_timestep,
        "final_window_mean_local_linear_velocity": np.mean(
            final_window_local_velocities, axis=0
        ).tolist(),
        "final_window_rms_local_linear_speed": float(
            np.sqrt(
                np.mean(
                    np.sum(np.square(final_window_local_velocities), axis=1)
                )
            )
        ),
        "final_qpos_sha256": array_digest(data.qpos),
        "final_qvel_sha256": array_digest(data.qvel),
        "freeze_phase_at_zero": config.freeze_phase_at_zero,
        "initial_position": initial_position.tolist(),
        "initial_yaw": initial_yaw,
        "horizontal_path_length": horizontal_path_length,
        "hold_pose_at_zero": config.hold_pose_at_zero,
        "mean_local_linear_velocity": np.mean(local_velocities, axis=0).tolist(),
        "mean_local_linear_velocity_first_quarter": np.mean(
            local_velocities[:quarter_steps], axis=0
        ).tolist(),
        "mean_local_linear_velocity_last_quarter": np.mean(
            local_velocities[-quarter_steps:], axis=0
        ).tolist(),
        "linear_velocity_tracking_rmse": float(
            np.sqrt(np.mean(np.square(linear_velocity_errors)))
        ),
        "maximum_contacts": maximum_contacts,
        "minimum_root_height": minimum_root_height,
        "mujoco_version": mujoco.__version__,
        "onnxruntime_version": version("onnxruntime"),
        "passed": all(checks.values()),
        "playground_revision": PLAYGROUND_REVISION,
        "policy_bytes": policy_path.stat().st_size,
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "seed": config.seed,
        "simulation_steps": simulation_steps,
        "simulation_time": float(data.time),
        "simulation_timestep": simulation_timestep,
        "stop_at": config.stop_at,
        "stop_applied_at": stop_applied_at,
        "post_stop_path_length": post_stop_path_length,
        "post_stop_displacement": (
            (final_position - stop_position).tolist()
            if stop_position is not None
            else None
        ),
        "yaw_rotation": accumulated_yaw,
        "mean_yaw_rate": accumulated_yaw / config.duration,
        "mean_yaw_rate_sensor": float(np.mean(yaw_rates)),
        "yaw_rate_tracking_rmse": float(
            np.sqrt(np.mean(np.square(yaw_rate_errors)))
        ),
    }


def main(config: Config) -> None:
    result = run_reproduction(config)
    serialized_result = json.dumps(result, sort_keys=True)

    print(serialized_result)

    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(
            serialized_result + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
