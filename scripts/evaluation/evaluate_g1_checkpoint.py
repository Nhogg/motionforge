"""evaluate_g1_checkpoint.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Evaluate a trained G1 checkpoint on deterministic command scenarios.

    The evaluator restores one Brax PPO checkpoint and runs fixed standing,
    walking, turning, and stopping commands.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
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
        "logs/p2/training/g1_standing_full_seed0/checkpoints/000191692800"
    )
    seed: int = 0
    seeds: int = 4
    naconmax: int = 16
    njmax: int = 128
    duration: float = 10.0
    stop_at: float = 5.0
    output: Path = Path("logs/p2/g1_checkpoint_evaluation_best_a.json")


class Scenario(NamedTuple):
    name: str
    initial_command: tuple[float, float, float]
    final_command: tuple[float, float, float]
    switch_time: float | None = None


SCENARIOS = (
    Scenario("stand", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    Scenario("forward", (0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
    Scenario("backward", (-0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)),
    Scenario("lateral_left", (0.0, 0.3, 0.0), (0.0, 0.3, 0.0)),
    Scenario("lateral_right", (0.0, -0.3, 0.0), (0.0, -0.3, 0.0)),
    Scenario("turn_left", (0.0, 0.0, 0.5), (0.0, 0.0, 0.5)),
    Scenario("turn_right", (0.0, 0.0, -0.5), (0.0, 0.0, -0.5)),
    Scenario(
        "curve_left",
        (0.3, 0.0, 0.5),
        (0.3, 0.0, 0.5),
    ),
    Scenario(
        "curve_right",
        (0.3, 0.0, -0.5),
        (0.3, 0.0, -0.5),
    ),
    Scenario(
        "stop",
        (0.5, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        switch_time=5.0,
    ),
)


def force_command(state, command: jax.Array):
    """Put a command into both environment state and policy observation."""
    info = dict(state.info)
    info["command"] = command

    observation = dict(state.obs)
    observation["state"] = (
        observation["state"].at[COMMAND_OBSERVATION_SLICE].set(command)
    )
    observation["privileged_state"] = (
        observation["privileged_state"].at[COMMAND_OBSERVATION_SLICE].set(command)
    )

    return state.replace(info=info, obs=observation)


def validate_config(config: Config) -> None:
    if config.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if config.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if not 0.0 < config.stop_at < config.duration:
        raise ValueError("--stop-at must lie within the rollout duration")


def main(config: Config) -> None:
    validate_config(config)

    checkpoint_path = config.checkpoint.resolve()
    network_config_path = checkpoint_path / "ppo_network_config.json"

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    ppo_network = load_ppo_network(network_config_path)
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

    environment = G1StandingJoystick(config=environment_config)

    control_timestep = float(environment.dt)
    rollout_steps = round(config.duration / control_timestep)
    stop_step = round(config.stop_at / control_timestep)

    @jax.jit
    def controlled_step(state, command, action_rng):
        commanded_state = force_command(state, command)
        action, _ = policy(commanded_state.obs, action_rng)
        next_state = environment.step(commanded_state, action)

        local_velocity = environment.get_local_linvel(
            next_state.data,
            "pelvis",
        )
        angular_velocity = environment.get_gyro(
            next_state.data,
            "pelvis",
        )

        return next_state, local_velocity, angular_velocity

    scenario_results: list[dict[str, object]] = []
    all_finite = True
    all_rollouts_completed = True

    for scenario in SCENARIOS:
        seed_results: list[dict[str, object]] = []

        scenario_switch_step = (
            round(scenario.switch_time / control_timestep)
            if scenario.switch_time is not None
            else None
        )

        for seed_offset in range(config.seeds):
            rollout_seed = config.seed + seed_offset
            rng = jax.random.PRNGKey(rollout_seed)
            rng, reset_rng = jax.random.split(rng)
            state = environment.reset(reset_rng)

            terminated = jp.array(False)
            steps_alive = jp.array(0)
            total_reward = jp.array(0.0)
            minimum_root_height = state.data.qpos[2]

            sum_local_velocity = jp.zeros(2)
            sum_yaw_rate = jp.array(0.0)
            sum_linear_squared_error = jp.array(0.0)
            sum_yaw_squared_error = jp.array(0.0)

            pre_switch_velocity_sum = jp.zeros(2)
            pre_switch_steps = jp.array(0)
            post_switch_velocity_sum = jp.zeros(2)
            post_switch_steps = jp.array(0)

            maximum_foot_height = jp.full((2,), -jp.inf)
            minimum_foot_height = jp.full((2,), jp.inf)
            foot_contact_steps = jp.zeros(2, dtype=jp.int32)
            foot_contact_transitions = jp.zeros(2, dtype=jp.int32)
            double_support_steps = jp.array(0, dtype=jp.int32)
            flight_steps = jp.array(0, dtype=jp.int32)
            previous_contact = jp.asarray(state.info["last_contact"], dtype=bool)

            for step in range(rollout_steps):
                use_final_command = (
                    scenario_switch_step is not None and step >= scenario_switch_step
                )
                command_tuple = (
                    scenario.final_command
                    if use_final_command
                    else scenario.initial_command
                )
                command = jp.asarray(command_tuple)

                rng, action_rng = jax.random.split(rng)
                state, local_velocity, angular_velocity = controlled_step(
                    state,
                    command,
                    action_rng,
                )

                valid = ~terminated
                valid_float = valid.astype(jp.float32)

                contact = jp.asarray(
                    state.info["last_contact"],
                    dtype=bool,
                )
                foot_height = state.data.site_xpos[
                    environment._feet_site_id,
                    2,
                ]

                maximum_foot_height = jp.maximum(
                    maximum_foot_height,
                    jp.where(valid, foot_height, -jp.inf),
                )
                minimum_foot_height = jp.minimum(
                    minimum_foot_height,
                    jp.where(valid, foot_height, jp.inf),
                )

                foot_contact_steps += (contact & valid).astype(jp.int32)
                foot_contact_transitions += (
                    (contact != previous_contact) & valid
                ).astype(jp.int32)

                double_support_steps += (valid & jp.all(contact)).astype(jp.int32)
                flight_steps += (valid & ~jp.any(contact)).astype(jp.int32)

                previous_contact = jp.where(
                    valid,
                    contact,
                    previous_contact,
                )

                linear_error = local_velocity[:2] - command[:2]
                yaw_error = angular_velocity[2] - command[2]

                steps_alive += valid.astype(jp.int32)
                total_reward += state.reward * valid_float
                sum_local_velocity += local_velocity[:2] * valid_float
                sum_yaw_rate += angular_velocity[2] * valid_float
                sum_linear_squared_error += (
                    jp.sum(jp.square(linear_error)) * valid_float
                )
                sum_yaw_squared_error += jp.square(yaw_error) * valid_float
                minimum_root_height = jp.minimum(
                    minimum_root_height,
                    jp.where(
                        valid,
                        state.data.qpos[2],
                        minimum_root_height,
                    ),
                )

                if scenario_switch_step is not None:
                    if step < scenario_switch_step:
                        pre_switch_velocity_sum += local_velocity[:2] * valid_float
                        pre_switch_steps += valid.astype(jp.int32)
                    else:
                        post_switch_velocity_sum += local_velocity[:2] * valid_float
                        post_switch_steps += valid.astype(jp.int32)

                terminated |= state.done.astype(bool)

            steps_alive_value = int(np.asarray(steps_alive))
            denominator = max(steps_alive_value, 1)

            mean_local_velocity = np.asarray(sum_local_velocity / denominator)
            mean_yaw_rate = float(np.asarray(sum_yaw_rate / denominator))

            rollout_finite = bool(
                np.isfinite(np.asarray(state.data.qpos)).all()
                and np.isfinite(np.asarray(state.data.qvel)).all()
                and np.isfinite(mean_local_velocity).all()
                and np.isfinite(mean_yaw_rate)
                and np.isfinite(np.asarray(total_reward))
            )

            maximum_foot_height_array = np.asarray(maximum_foot_height)
            minimum_foot_height_array = np.asarray(minimum_foot_height)

            seed_result: dict[str, object] = {
                "linear_velocity_tracking_rmse": float(
                    np.sqrt(np.asarray(sum_linear_squared_error / denominator))
                ),
                "mean_local_linear_velocity": (mean_local_velocity.tolist()),
                "mean_yaw_rate": mean_yaw_rate,
                "minimum_root_height": float(np.asarray(minimum_root_height)),
                "rollout_finite": rollout_finite,
                "seed": rollout_seed,
                "steps_alive": steps_alive_value,
                "survived": steps_alive_value == rollout_steps,
                "total_reward": float(np.asarray(total_reward)),
                "yaw_rate_tracking_rmse": float(
                    np.sqrt(np.asarray(sum_yaw_squared_error / denominator))
                ),
                "double_support_fraction": float(np.asarray(double_support_steps))
                / denominator,
                "flight_fraction": float(np.asarray(flight_steps)) / denominator,
                "foot_contact_duty_factor": (
                    np.asarray(foot_contact_steps) / denominator
                ).tolist(),
                "foot_contact_transitions": np.asarray(
                    foot_contact_transitions
                ).tolist(),
                "foot_height_range": (
                    maximum_foot_height_array - minimum_foot_height_array
                ).tolist(),
                "maximum_foot_height": (maximum_foot_height_array.tolist()),
            }

            if scenario_switch_step is not None:
                pre_denominator = max(
                    int(np.asarray(pre_switch_steps)),
                    1,
                )
                post_denominator = max(
                    int(np.asarray(post_switch_steps)),
                    1,
                )
                seed_result["mean_velocity_before_switch"] = np.asarray(
                    pre_switch_velocity_sum / pre_denominator
                ).tolist()
                seed_result["mean_velocity_after_switch"] = np.asarray(
                    post_switch_velocity_sum / post_denominator
                ).tolist()

            seed_results.append(seed_result)
            all_finite &= rollout_finite
            all_rollouts_completed &= steps_alive_value == rollout_steps

        scenario_results.append(
            {
                "final_command": list(scenario.final_command),
                "initial_command": list(scenario.initial_command),
                "name": scenario.name,
                "rollouts": seed_results,
                "survival_rate": float(
                    np.mean([result["survived"] for result in seed_results])
                ),
                "switch_time": scenario.switch_time,
            }
        )

    checks = {
        "all_rollouts_completed": all_rollouts_completed,
        "finite_values": all_finite,
        "gpu_backend": jax.default_backend() == "gpu",
        "parameters_loaded": len(parameters) == 3,
        "scenario_count": len(scenario_results) == len(SCENARIOS),
    }

    result = {
        "backend": jax.default_backend(),
        "checkpoint": str(checkpoint_path),
        "checks": checks,
        "control_timestep": control_timestep,
        "duration": config.duration,
        "experiment": "g1_checkpoint_command_evaluation",
        "jax_version": jax.__version__,
        "mujoco_version": version("mujoco"),
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "rollout_steps": rollout_steps,
        "scenario_count": len(SCENARIOS),
        "scenarios": scenario_results,
        "seed": config.seed,
        "seeds": config.seeds,
        "naconmax": config.naconmax,
        "njmax": config.njmax,
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
