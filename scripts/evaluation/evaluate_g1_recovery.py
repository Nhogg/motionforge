"""Evaluate deterministic planar push recovery for a trained G1 checkpoint.

Inputs are a Brax PPO checkpoint, rollout seeds, a fixed standing command, and
the time and magnitude of a one-shot root-velocity disturbance. The script
runs four world-frame push directions with random environment pushes disabled.
It writes one machine-readable JSON report containing survival, posture,
tracking, and sustained-recovery-time measurements. This is an acceptance
evaluator only: it does not train or modify the policy.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
import tyro
from brax.training import checkpoint
from brax.training.agents.ppo import networks as ppo_networks

from motionforge.compat.brax_checkpoint import load_ppo_network
from motionforge.envs.g1_standing import G1StandingJoystick, default_config

COMMAND_OBSERVATION_SLICE = slice(9, 12)


@dataclass(frozen=True)
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 0
    seeds: int = 4
    duration: float = 10.0
    push_at: float = 3.0
    push_speed: float = 1.0
    recovery_speed_threshold: float = 0.15
    recovery_hold_seconds: float = 0.5
    naconmax: int = 16
    njmax: int = 128
    output: Path = Path("logs/p2/g1_recovery_1ms_a.json")


class PushScenario(NamedTuple):
    name: str
    velocity_delta: tuple[float, float]


class ContactTolerantG1StandingJoystick(G1StandingJoystick):
    """Diagnostic environment that terminates only on physical failure."""

    def _get_termination(self, data):
        return (
            (self.get_gravity(data, "torso")[-1] < 0.0)
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
        )


def force_command(state, command: jax.Array):
    """Put a command into both environment state and policy observations."""
    info = dict(state.info)
    info["command"] = command

    observation = dict(state.obs)
    observation["state"] = (
        observation["state"].at[COMMAND_OBSERVATION_SLICE].set(command)
    )
    observation["privileged_state"] = (
        observation["privileged_state"]
        .at[COMMAND_OBSERVATION_SLICE]
        .set(command)
    )
    return state.replace(info=info, obs=observation)


def sustained_recovery_time(
    errors: np.ndarray,
    push_step: int,
    hold_steps: int,
    threshold: float,
    control_timestep: float,
) -> float | None:
    """Return time after the push when error stays low for the hold window."""
    recovered = errors <= threshold
    final_start = len(recovered) - hold_steps
    for start in range(push_step, final_start + 1):
        if bool(np.all(recovered[start : start + hold_steps])):
            return (start - push_step + 1) * control_timestep
    return None


def validate_config(config: Config) -> None:
    if config.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if config.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if not 0.0 < config.push_at < config.duration:
        raise ValueError("--push-at must lie within the rollout duration")
    if config.push_speed <= 0.0:
        raise ValueError("--push-speed must be positive")
    if config.recovery_speed_threshold <= 0.0:
        raise ValueError("--recovery-speed-threshold must be positive")
    if config.recovery_hold_seconds <= 0.0:
        raise ValueError("--recovery-hold-seconds must be positive")


def main(config: Config) -> None:
    validate_config(config)

    checkpoint_path = config.checkpoint.resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    ppo_network = load_ppo_network(checkpoint_path / "ppo_network_config.json")
    parameters = checkpoint.load(checkpoint_path)
    policy = ppo_networks.make_inference_fn(ppo_network)(
        parameters,
        deterministic=True,
    )

    environment_config = default_config()
    environment_config.impl = "warp"
    environment_config.naconmax = config.naconmax
    environment_config.njmax = config.njmax
    environment_config.push_config.enable = False
    environment = ContactTolerantG1StandingJoystick(config=environment_config)

    control_timestep = float(environment.dt)
    rollout_steps = round(config.duration / control_timestep)
    push_step = round(config.push_at / control_timestep)
    hold_steps = max(1, round(config.recovery_hold_seconds / control_timestep))
    if push_step + hold_steps > rollout_steps:
        raise ValueError("Recovery hold window does not fit after --push-at")

    magnitude = config.push_speed
    scenarios = (
        PushScenario("forward", (magnitude, 0.0)),
        PushScenario("backward", (-magnitude, 0.0)),
        PushScenario("left", (0.0, magnitude)),
        PushScenario("right", (0.0, -magnitude)),
    )
    standing_command = jp.zeros(3)
    prohibited_contact_sensor_addresses = jp.asarray(
        [
            environment._mj_model.sensor_adr[
                environment._right_foot_left_foot_found_sensor
            ],
            environment._mj_model.sensor_adr[
                environment._left_foot_right_shin_found_sensor
            ],
            environment._mj_model.sensor_adr[
                environment._right_foot_left_shin_found_sensor
            ],
        ]
    )

    @jax.jit
    def controlled_step(state, action_rng):
        commanded_state = force_command(state, standing_command)
        action, _ = policy(commanded_state.obs, action_rng)
        next_state = environment.step(commanded_state, action)
        local_velocity = environment.get_local_linvel(next_state.data, "pelvis")
        torso_gravity = environment.get_gravity(next_state.data, "torso")
        prohibited_contacts = (
            next_state.data.sensordata[prohibited_contact_sensor_addresses] > 0
        )
        return next_state, local_velocity, torso_gravity, prohibited_contacts

    scenario_results: list[dict[str, object]] = []
    all_finite = True
    all_rollouts_completed = True
    all_task_rollouts_completed = True

    for scenario in scenarios:
        rollout_results: list[dict[str, object]] = []
        velocity_delta = jp.asarray(scenario.velocity_delta)

        for seed_offset in range(config.seeds):
            rollout_seed = config.seed + seed_offset
            rng = jax.random.PRNGKey(rollout_seed)
            rng, reset_rng = jax.random.split(rng)
            state = environment.reset(reset_rng)

            terminated = False
            steps_alive = 0
            minimum_root_height = float(np.asarray(state.data.qpos[2]))
            maximum_tilt_radians = 0.0
            local_speed_errors: list[float] = []
            termination_causes: list[str] = []
            termination_time: float | None = None
            first_prohibited_contact_time: float | None = None

            for step in range(rollout_steps):
                if step == push_step:
                    qvel = state.data.qvel.at[:2].add(velocity_delta)
                    state = state.replace(data=state.data.replace(qvel=qvel))

                rng, action_rng = jax.random.split(rng)
                state, local_velocity, torso_gravity, prohibited_contacts = (
                    controlled_step(
                        state,
                        action_rng,
                    )
                )

                if not terminated:
                    velocity = np.asarray(local_velocity[:2])
                    gravity = np.asarray(torso_gravity)
                    speed_error = float(np.linalg.norm(velocity))
                    # Playground's framequat gravity sensor reports +Z when
                    # the torso frame is upright; its fall condition likewise
                    # triggers when this component becomes negative.
                    upright_cosine = float(np.clip(gravity[2], -1.0, 1.0))

                    local_speed_errors.append(speed_error)
                    steps_alive += 1
                    minimum_root_height = min(
                        minimum_root_height,
                        float(np.asarray(state.data.qpos[2])),
                    )
                    maximum_tilt_radians = max(
                        maximum_tilt_radians,
                        float(np.arccos(upright_cosine)),
                    )

                done_now = bool(np.asarray(state.done))
                contact_flags = np.asarray(prohibited_contacts)
                if (
                    first_prohibited_contact_time is None
                    and bool(np.any(contact_flags))
                ):
                    first_prohibited_contact_time = (step + 1) * control_timestep
                if done_now and not terminated:
                    gravity = np.asarray(torso_gravity)
                    if gravity[2] < 0.0:
                        termination_causes.append("torso_inverted")
                    if not (
                        np.isfinite(np.asarray(state.data.qpos)).all()
                        and np.isfinite(np.asarray(state.data.qvel)).all()
                    ):
                        termination_causes.append("non_finite_state")
                    if not termination_causes:
                        termination_causes.append("unclassified")
                    termination_time = (step + 1) * control_timestep
                terminated = terminated or done_now

            errors = np.asarray(local_speed_errors)
            post_push_errors = errors[push_step:]
            recovery_time = sustained_recovery_time(
                errors,
                push_step,
                hold_steps,
                config.recovery_speed_threshold,
                control_timestep,
            )
            rollout_finite = bool(
                np.isfinite(np.asarray(state.data.qpos)).all()
                and np.isfinite(np.asarray(state.data.qvel)).all()
                and np.isfinite(errors).all()
                and np.isfinite(minimum_root_height)
                and np.isfinite(maximum_tilt_radians)
            )

            rollout_results.append(
                {
                    "maximum_post_push_speed_error": float(
                        np.max(post_push_errors)
                    ),
                    "maximum_tilt_degrees": float(
                        np.degrees(maximum_tilt_radians)
                    ),
                    "minimum_root_height": minimum_root_height,
                    "post_push_speed_rmse": float(
                        np.sqrt(np.mean(np.square(post_push_errors)))
                    ),
                    "recovered": recovery_time is not None,
                    "recovery_time": recovery_time,
                    "rollout_finite": rollout_finite,
                    "seed": rollout_seed,
                    "steps_alive": steps_alive,
                    "survived": steps_alive == rollout_steps,
                    "first_prohibited_contact_time": (
                        first_prohibited_contact_time
                    ),
                    "prohibited_contact_after_push": bool(
                        first_prohibited_contact_time is not None
                        and first_prohibited_contact_time >= config.push_at
                    ),
                    "task_survived": first_prohibited_contact_time is None,
                    "termination_causes": termination_causes,
                    "termination_time": termination_time,
                    "time_from_push_to_termination": (
                        termination_time - config.push_at
                        if termination_time is not None
                        else None
                    ),
                }
            )
            all_finite &= rollout_finite
            all_rollouts_completed &= steps_alive == rollout_steps
            all_task_rollouts_completed &= first_prohibited_contact_time is None

        recovered_times = [
            result["recovery_time"]
            for result in rollout_results
            if result["recovery_time"] is not None
        ]
        scenario_results.append(
            {
                "mean_recovery_time": (
                    float(np.mean(recovered_times)) if recovered_times else None
                ),
                "name": scenario.name,
                "recovery_rate": float(
                    np.mean([result["recovered"] for result in rollout_results])
                ),
                "rollouts": rollout_results,
                "survival_rate": float(
                    np.mean([result["survived"] for result in rollout_results])
                ),
                "task_survival_rate": float(
                    np.mean(
                        [result["task_survived"] for result in rollout_results]
                    )
                ),
                "velocity_delta": list(scenario.velocity_delta),
            }
        )

    checks = {
        "all_rollouts_completed": all_rollouts_completed,
        "finite_values": all_finite,
        "gpu_backend": jax.default_backend() == "gpu",
        "parameters_loaded": len(parameters) == 3,
        "scenario_count": len(scenario_results) == len(scenarios),
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
        "experiment": "g1_checkpoint_planar_push_recovery",
        "contact_tolerant": True,
        "jax_version": jax.__version__,
        "mujoco_version": version("mujoco"),
        "passed": all(checks.values()),
        "push_step": push_step,
        "python_version": platform.python_version(),
        "recovery_hold_steps": hold_steps,
        "rollout_steps": rollout_steps,
        "scenario_count": len(scenarios),
        "scenarios": scenario_results,
        "task_rollouts_completed": all_task_rollouts_completed,
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
    main(tyro.cli(Config))
