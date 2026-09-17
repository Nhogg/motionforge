"""Evaluate the accepted G1 controller under abrupt command transitions.

The evaluator restores one deterministic P2 checkpoint and applies fixed,
repeatable schedules containing reversals, braking, lateral switches, and yaw
switches. It records physical survival, tracking error, posture excursions,
and transition recovery to a machine-readable JSON artifact. Random pushes are
disabled so command changes are the only intentional disturbance.
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from motionforge.cli import run_hydra
from motionforge.controllers import G1LocomotionController, apply_velocity_command
from motionforge.envs.g1_standing import G1StandingJoystick, default_config


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    seeds: int = 4
    duration: float = 10.0
    segment_duration: float = 2.0
    recovery_hold_seconds: float = 0.2
    recovery_linear_error: float = 0.25
    recovery_yaw_error: float = 0.30
    naconmax: int = 16
    njmax: int = 128
    output: Path = Path("logs/p4/g1_rapid_commands_a.json")


class Scenario(NamedTuple):
    name: str
    commands: tuple[tuple[float, float, float], ...]


SCENARIOS = (
    Scenario("forward_reverse", ((0.5, 0.0, 0.0), (-0.5, 0.0, 0.0))),
    Scenario("lateral_switch", ((0.0, 0.25, 0.0), (0.0, -0.25, 0.0))),
    Scenario("yaw_switch", ((0.0, 0.0, 0.6), (0.0, 0.0, -0.6))),
    Scenario(
        "diagonal_reverse",
        ((0.4, 0.2, 0.4), (-0.4, -0.2, -0.4)),
    ),
    Scenario("brake_restart", ((0.5, 0.0, 0.0), (0.0, 0.0, 0.0))),
)


class PhysicalOnlyG1StandingJoystick(G1StandingJoystick):
    """Ignore incidental leg contacts while retaining physical fall checks."""

    def _get_termination(self, data):
        return (
            (self.get_gravity(data, "torso")[-1] < 0.0)
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
        )


def _validate_config(config: Config) -> None:
    if config.seeds <= 0:
        raise ValueError("seeds must be positive")
    if config.duration <= 0.0 or config.segment_duration <= 0.0:
        raise ValueError("duration and segment_duration must be positive")
    if config.recovery_hold_seconds <= 0.0:
        raise ValueError("recovery_hold_seconds must be positive")
    if config.recovery_linear_error <= 0.0 or config.recovery_yaw_error <= 0.0:
        raise ValueError("recovery error thresholds must be positive")


def main(config: Config) -> None:
    _validate_config(config)
    checkpoint_path = config.checkpoint.resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    environment_config = default_config()
    environment_config.impl = "warp"
    environment_config.naconmax = config.naconmax
    environment_config.njmax = config.njmax
    environment_config.push_config.enable = False
    environment = PhysicalOnlyG1StandingJoystick(config=environment_config)
    controller = G1LocomotionController.from_checkpoint(
        checkpoint_path,
        environment,
    )

    control_timestep = float(environment.dt)
    rollout_steps = round(config.duration / control_timestep)
    segment_steps = round(config.segment_duration / control_timestep)
    recovery_hold_steps = round(config.recovery_hold_seconds / control_timestep)

    if not math.isclose(
        segment_steps * control_timestep,
        config.segment_duration,
        abs_tol=1e-9,
    ):
        raise ValueError("segment_duration must align with the control timestep")

    @jax.jit
    def controlled_step(state, command, action_rng):
        commanded_state = apply_velocity_command(state, command)
        output = controller.act(commanded_state.obs, action_rng)
        next_state = environment.step(
            commanded_state,
            output.normalized_action,
        )
        return (
            next_state,
            environment.get_local_linvel(next_state.data, "pelvis")[:2],
            environment.get_gyro(next_state.data, "pelvis")[2],
            environment.get_gravity(next_state.data, "torso")[-1],
        )

    scenario_results = []
    all_finite = True
    all_survived = True

    for scenario in SCENARIOS:
        rollouts = []
        for seed_offset in range(config.seeds):
            rollout_seed = config.seed + seed_offset
            rng = jax.random.PRNGKey(rollout_seed)
            rng, reset_rng = jax.random.split(rng)
            state = environment.reset(reset_rng)

            squared_linear_errors = []
            squared_yaw_errors = []
            minimum_root_height = float(np.asarray(state.data.qpos[2]))
            minimum_up_alignment = 1.0
            transitions = []
            pending_transition = None
            recovery_streak = 0
            rollout_finite = True
            survived = True
            steps_alive = 0

            for step_index in range(rollout_steps):
                segment_index = step_index // segment_steps
                command_index = segment_index % len(scenario.commands)
                command = jp.asarray(
                    scenario.commands[command_index],
                    dtype=jp.float32,
                )

                if step_index > 0 and step_index % segment_steps == 0:
                    if pending_transition is not None:
                        transitions.append(
                            {
                                "recovered": False,
                                "recovery_time": None,
                                "step": pending_transition,
                            }
                        )
                    pending_transition = step_index
                    recovery_streak = 0

                rng, action_rng = jax.random.split(rng)
                state, local_velocity, yaw_rate, up_alignment = controlled_step(
                    state,
                    command,
                    action_rng,
                )
                state.data.qpos.block_until_ready()

                local_velocity_host = np.asarray(local_velocity)
                yaw_rate_host = float(np.asarray(yaw_rate))
                up_alignment_host = float(np.asarray(up_alignment))
                root_height = float(np.asarray(state.data.qpos[2]))
                state_finite = bool(
                    np.isfinite(np.asarray(state.data.qpos)).all()
                    and np.isfinite(np.asarray(state.data.qvel)).all()
                )
                rollout_finite &= state_finite
                minimum_root_height = min(minimum_root_height, root_height)
                minimum_up_alignment = min(
                    minimum_up_alignment,
                    up_alignment_host,
                )

                linear_error = local_velocity_host - np.asarray(command[:2])
                yaw_error = yaw_rate_host - float(np.asarray(command[2]))
                squared_linear_errors.append(float(np.dot(linear_error, linear_error)))
                squared_yaw_errors.append(yaw_error * yaw_error)
                steps_alive = step_index + 1

                if pending_transition is not None:
                    recovered_now = (
                        np.linalg.norm(linear_error) <= config.recovery_linear_error
                        and abs(yaw_error) <= config.recovery_yaw_error
                    )
                    recovery_streak = recovery_streak + 1 if recovered_now else 0
                    if recovery_streak >= recovery_hold_steps:
                        first_recovered_step = step_index - recovery_hold_steps + 1
                        transitions.append(
                            {
                                "recovered": True,
                                "recovery_time": (
                                    first_recovered_step - pending_transition
                                )
                                * control_timestep,
                                "step": pending_transition,
                            }
                        )
                        pending_transition = None
                        recovery_streak = 0

                if bool(np.asarray(state.done)):
                    survived = False
                    break

            if pending_transition is not None:
                transitions.append(
                    {
                        "recovered": False,
                        "recovery_time": None,
                        "step": pending_transition,
                    }
                )

            recovered_transitions = [
                transition for transition in transitions if transition["recovered"]
            ]
            recovery_times = [
                transition["recovery_time"] for transition in recovered_transitions
            ]
            maximum_tilt_degrees = math.degrees(
                math.acos(float(np.clip(minimum_up_alignment, -1.0, 1.0)))
            )
            rollouts.append(
                {
                    "linear_tracking_rmse": math.sqrt(
                        sum(squared_linear_errors) / len(squared_linear_errors)
                    ),
                    "maximum_tilt_degrees": maximum_tilt_degrees,
                    "mean_recovery_time": (
                        sum(recovery_times) / len(recovery_times)
                        if recovery_times
                        else None
                    ),
                    "minimum_root_height": minimum_root_height,
                    "rollout_finite": rollout_finite,
                    "seed": rollout_seed,
                    "steps_alive": steps_alive,
                    "survived": survived,
                    "transition_recovery_rate": (
                        len(recovered_transitions) / len(transitions)
                        if transitions
                        else 1.0
                    ),
                    "transitions": transitions,
                    "yaw_tracking_rmse": math.sqrt(
                        sum(squared_yaw_errors) / len(squared_yaw_errors)
                    ),
                }
            )
            all_finite &= rollout_finite
            all_survived &= survived

        scenario_results.append(
            {
                "name": scenario.name,
                "rollouts": rollouts,
                "survival_rate": sum(rollout["survived"] for rollout in rollouts)
                / len(rollouts),
                "transition_recovery_rate": sum(
                    rollout["transition_recovery_rate"] for rollout in rollouts
                )
                / len(rollouts),
            }
        )

    checks = {
        "all_rollouts_finite": all_finite,
        "all_rollouts_survived": all_survived,
        "all_transitions_recovered": all(
            scenario["transition_recovery_rate"] == 1.0 for scenario in scenario_results
        ),
        "gpu_backend": jax.default_backend() == "gpu",
        "scenario_count": len(scenario_results) == len(SCENARIOS),
    }
    result = {
        "backend": jax.default_backend(),
        "checkpoint": str(checkpoint_path),
        "checks": checks,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "output": str(config.output),
        },
        "control_timestep": control_timestep,
        "experiment": "g1_rapid_command_evaluation",
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "scenarios": scenario_results,
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
