from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.controllers.g1 import (
    G1LocomotionController,
    G1VelocityCommand,
    apply_velocity_command,
)
from motionforge.envs.g1_standing import G1StandingJoystick, default_config


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 0
    command_x: float = 0.3
    command_y: float = -0.2
    command_yaw: float = 0.5
    naconmax: int = 16
    njmax: int = 128
    output: Path = Path("logs/p2/g1_controller_interface_a.json")


def main(config: Config) -> None:
    environment_config = default_config()
    environment_config.impl = "warp"
    environment_config.naconmax = config.naconmax
    environment_config.njmax = config.njmax
    environment_config.push_config.enable = False
    environment = G1StandingJoystick(config=environment_config)

    controller = G1LocomotionController.from_checkpoint(
        config.checkpoint,
        environment,
    )
    command = G1VelocityCommand(
        config.command_x,
        config.command_y,
        config.command_yaw,
    ).as_array()

    rng = jax.random.PRNGKey(config.seed)
    rng, reset_rng, action_rng = jax.random.split(rng, 3)
    state = environment.reset(reset_rng)
    commanded_state = apply_velocity_command(state, command)

    @jax.jit
    def controller_call(observation, policy_rng):
        return controller.act(observation, policy_rng)

    output_a = controller_call(commanded_state.obs, action_rng)
    output_b = controller_call(commanded_state.obs, action_rng)
    next_state = environment.step(commanded_state, output_a.normalized_action)

    normalized_action = np.asarray(output_a.normalized_action)
    joint_targets = np.asarray(output_a.joint_position_targets)
    expected_targets = np.asarray(
        environment._default_pose
        + output_a.normalized_action * environment._config.action_scale
    )
    state_command = np.asarray(commanded_state.info["command"])
    policy_command = np.asarray(commanded_state.obs["state"][9:12])

    checks = {
        "action_finite": bool(np.isfinite(normalized_action).all()),
        "action_shape": normalized_action.shape == (29,),
        "command_round_trip": bool(
            np.array_equal(state_command, np.asarray(command))
            and np.array_equal(policy_command, np.asarray(command))
        ),
        "deterministic_action": bool(
            np.array_equal(
                normalized_action,
                np.asarray(output_b.normalized_action),
            )
        ),
        "environment_stepped": bool(float(np.asarray(next_state.data.time)) > 0.0),
        "gpu_backend": jax.default_backend() == "gpu",
        "joint_targets_match_environment": bool(
            np.allclose(joint_targets, expected_targets, rtol=0.0, atol=0.0)
        ),
        "joint_targets_shape": joint_targets.shape == (29,),
        "state_finite": bool(
            np.isfinite(np.asarray(next_state.data.qpos)).all()
            and np.isfinite(np.asarray(next_state.data.qvel)).all()
        ),
    }
    result = {
        "action_shape": list(normalized_action.shape),
        "backend": jax.default_backend(),
        "checkpoint": str(controller.checkpoint_path),
        "checks": checks,
        "command": np.asarray(command).tolist(),
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "output": str(config.output),
        },
        "control_timestep": float(environment.dt),
        "experiment": "g1_controller_interface_smoke",
        "jax_version": jax.__version__,
        "joint_target_shape": list(joint_targets.shape),
        "mujoco_version": version("mujoco"),
        "passed": all(checks.values()),
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
