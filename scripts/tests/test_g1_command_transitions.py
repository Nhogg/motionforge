"""Validate deterministic rapid-command resampling and reversal bookkeeping."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs.g1_standing import G1StandingJoystick, default_config


@dataclass
class Config:
    seed: int = 0
    transition_interval: float = 0.04
    output: Path = Path("logs/p4/g1_command_transitions_a.json")


def main(config: Config) -> None:
    environment_config = default_config()
    environment_config.impl = "warp"
    environment_config.naconmax = 16
    environment_config.njmax = 128
    environment_config.push_config.enable = False
    environment_config.rapid_command_transitions = True
    environment_config.rapid_command_episode_probability = 1.0
    environment_config.command_transition_interval_min = config.transition_interval
    environment_config.command_transition_interval_max = config.transition_interval
    environment_config.command_reversal_probability = 1.0
    environment = G1StandingJoystick(config=environment_config)

    initial_command = jp.asarray([0.5, -0.25, 0.4])
    state = environment.reset(jax.random.PRNGKey(config.seed))
    info = dict(state.info)
    info["command"] = initial_command
    observation = dict(state.obs)
    for name in ("state", "privileged_state"):
        observation[name] = observation[name].at[9:12].set(initial_command)
    state = state.replace(info=info, obs=observation)

    step = jax.jit(environment.step)
    zero_action = jp.zeros(29)
    first = step(state, zero_action)
    second = step(first, zero_action)
    third = step(second, zero_action)
    fourth = step(third, zero_action)
    inactive_info = dict(state.info)
    inactive_info["rapid_command_episode"] = jp.asarray(False)
    inactive = state.replace(info=inactive_info)
    for _ in range(4):
        inactive = step(inactive, zero_action)
    fourth.data.qpos.block_until_ready()
    inactive.data.qpos.block_until_ready()

    initial_host = np.asarray(initial_command)
    first_host = np.asarray(first.info["command"])
    second_host = np.asarray(second.info["command"])
    third_host = np.asarray(third.info["command"])
    fourth_host = np.asarray(fourth.info["command"])
    state_observation_command = np.asarray(second.obs["state"][9:12])
    privileged_observation_command = np.asarray(second.obs["privileged_state"][9:12])

    checks = {
        "first_step_holds": bool(np.allclose(first_host, initial_host)),
        "second_step_reverses": bool(np.allclose(second_host, -initial_host)),
        "third_step_holds": bool(np.allclose(third_host, -initial_host)),
        "fourth_step_reverses_again": bool(np.allclose(fourth_host, initial_host)),
        "observation_command_updated": bool(
            np.allclose(state_observation_command, second_host)
            and np.allclose(privileged_observation_command, second_host)
        ),
        "state_finite": bool(
            np.isfinite(np.asarray(fourth.data.qpos)).all()
            and np.isfinite(np.asarray(fourth.data.qvel)).all()
        ),
        "transition_counter_resets": bool(
            int(np.asarray(second.info["command_transition_step"])) == 0
            and int(np.asarray(fourth.info["command_transition_step"])) == 0
        ),
        "transition_interval_exact": bool(
            int(np.asarray(second.info["command_transition_interval"])) == 2
        ),
        "rapid_episode_active": bool(np.asarray(second.info["rapid_command_episode"])),
        "inactive_episode_holds_command": bool(
            np.allclose(np.asarray(inactive.info["command"]), initial_host)
        ),
    }

    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "commands": {
            "initial": initial_host.tolist(),
            "step_1": first_host.tolist(),
            "step_2": second_host.tolist(),
            "step_3": third_host.tolist(),
            "step_4": fourth_host.tolist(),
        },
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "g1_rapid_command_transition_smoke",
        "jax_version": jax.__version__,
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "seed": config.seed,
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
