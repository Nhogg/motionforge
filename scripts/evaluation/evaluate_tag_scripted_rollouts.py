"""Audit complete scripted tag rollouts across deterministic reset seeds.

The evaluator runs the scripted pursuer and boundary-aware evader through the
accepted G1 locomotion checkpoint in the integrated two-agent MJX-Warp
environment. It writes per-seed terminal outcomes and integration diagnostics
to JSON. Rendering is intentionally separate from this quantitative audit.
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
from motionforge.controllers import (
    G1LocomotionController,
    build_g1_tag_policy_observation_layout,
    g1_tag_policy_observation,
)
from motionforge.envs import TagEnvironmentConfig, TwoG1TagEnvironment
from motionforge.envs.g1_standing import G1StandingJoystick, default_config
from motionforge.policies import (
    ScriptedEvaderConfig,
    ScriptedPursuerConfig,
    scripted_evader_command,
    scripted_pursuer_command,
)


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    seeds: int = 8
    separation: float = 2.0
    duration: float = 20.0
    naconmax: int = 64
    njmax: int = 256
    output: Path = Path("logs/p4/tag_scripted_rollouts_a.json")


def _validate_config(config: Config) -> None:
    if config.seeds <= 0:
        raise ValueError("seeds must be positive")
    if config.duration <= 0.0:
        raise ValueError("duration must be positive")
    if config.separation <= 0.0:
        raise ValueError("separation must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("contact capacities must be positive")


def main(config: Config) -> None:
    _validate_config(config)
    checkpoint_path = config.checkpoint.resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    source_config = default_config()
    source_config.impl = "warp"
    source_config.naconmax = 16
    source_config.njmax = 128
    source_config.push_config.enable = False
    source_environment = G1StandingJoystick(config=source_config)
    controller = G1LocomotionController.from_checkpoint(
        checkpoint_path,
        source_environment,
    )

    environment = TwoG1TagEnvironment(
        TagEnvironmentConfig(
            separation=config.separation,
            episode_duration=config.duration,
            naconmax=config.naconmax,
            njmax=config.njmax,
        )
    )
    observation_layout = build_g1_tag_policy_observation_layout(
        environment.model_bundle
    )
    pursuer_config = ScriptedPursuerConfig()
    evader_config = ScriptedEvaderConfig()
    default_pose = jp.asarray(source_environment._default_pose)
    action_scale = float(source_environment._config.action_scale)
    phase_dt = 2.0 * jp.pi * environment.config.control_timestep * 1.375
    rollout_steps = round(config.duration / environment.config.control_timestep)

    reset = jax.jit(environment.reset)

    @jax.jit
    def controlled_step(state, last_actions, phases, rng):
        commands = jp.stack(
            [
                scripted_pursuer_command(
                    state.observation.relative_position[0],
                    pursuer_config,
                ),
                scripted_evader_command(
                    state.observation.relative_position[1],
                    state.observation.arena_center_position[1],
                    evader_config,
                ),
            ]
        )
        rng, agent0_rng, agent1_rng = jax.random.split(rng, 3)
        observations = (
            g1_tag_policy_observation(
                state.data,
                observation_layout,
                0,
                commands[0],
                last_actions[0],
                phases[0],
                default_pose,
            ),
            g1_tag_policy_observation(
                state.data,
                observation_layout,
                1,
                commands[1],
                last_actions[1],
                phases[1],
                default_pose,
            ),
        )
        actions = jp.stack(
            [
                controller.act(observations[0], agent0_rng).normalized_action,
                controller.act(observations[1], agent1_rng).normalized_action,
            ]
        )
        joint_targets = default_pose + actions * action_scale
        next_state = environment.step(state, joint_targets)
        next_phases = jp.fmod(phases + phase_dt + jp.pi, 2.0 * jp.pi) - jp.pi
        return next_state, actions, next_phases, rng, commands, joint_targets

    deterministic_key = jax.random.PRNGKey(config.seed)
    deterministic_a = reset(deterministic_key)
    deterministic_b = reset(deterministic_key)
    deterministic_b.data.qpos.block_until_ready()
    deterministic_reset = bool(
        np.array_equal(
            np.asarray(deterministic_a.data.qpos),
            np.asarray(deterministic_b.data.qpos),
        )
        and np.array_equal(
            np.asarray(deterministic_a.data.qvel),
            np.asarray(deterministic_b.data.qvel),
        )
    )

    rollouts = []
    for seed_offset in range(config.seeds):
        rollout_seed = config.seed + seed_offset
        rng = jax.random.PRNGKey(rollout_seed)
        rng, reset_rng = jax.random.split(rng)
        state = reset(reset_rng)
        last_actions = jp.zeros((2, 29))
        phases = jp.asarray([[0.0, jp.pi], [0.0, jp.pi]])

        finite = True
        observation_consistent = True
        controls_finite = True
        minimum_root_height = float("inf")
        minimum_agent_distance = float("inf")
        maximum_command_magnitude = 0.0
        steps_executed = 0

        for _ in range(rollout_steps):
            state, last_actions, phases, rng, commands, joint_targets = controlled_step(
                state, last_actions, phases, rng
            )
            state.data.qpos.block_until_ready()
            steps_executed += 1

            qpos = np.asarray(state.data.qpos)
            qvel = np.asarray(state.data.qvel)
            command_host = np.asarray(commands)
            action_host = np.asarray(last_actions)
            target_host = np.asarray(joint_targets)
            relative_position = np.asarray(state.observation.relative_position)
            center_position = np.asarray(state.observation.arena_center_position)
            pelvis_position = np.stack(
                [
                    np.asarray(state.data.xpos[agent.root_body_id, :2])
                    for agent in environment.observation_layout.agents
                ]
            )
            root_height = np.asarray(state.diagnostics.root_height)

            finite &= bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
            controls_finite &= bool(
                np.isfinite(command_host).all()
                and np.isfinite(action_host).all()
                and np.isfinite(target_host).all()
            )
            observation_consistent &= bool(
                np.isfinite(relative_position).all()
                and np.isfinite(center_position).all()
                and np.allclose(
                    np.linalg.norm(relative_position, axis=1),
                    np.linalg.norm(pelvis_position[1] - pelvis_position[0]),
                    atol=1e-4,
                )
                and np.allclose(
                    np.linalg.norm(center_position, axis=1),
                    np.linalg.norm(pelvis_position, axis=1),
                    atol=1e-4,
                )
            )
            minimum_root_height = min(minimum_root_height, float(root_height.min()))
            minimum_agent_distance = min(
                minimum_agent_distance,
                float(np.linalg.norm(pelvis_position[1] - pelvis_position[0])),
            )
            maximum_command_magnitude = max(
                maximum_command_magnitude,
                float(np.linalg.norm(command_host, axis=1).max()),
            )

            if bool(np.asarray(state.done)):
                break

        termination = {
            "fallen": np.asarray(state.termination.fallen).tolist(),
            "out_of_bounds": np.asarray(state.termination.out_of_bounds).tolist(),
            "tagged": bool(np.asarray(state.termination.tagged)),
            "timed_out": bool(np.asarray(state.termination.timed_out)),
        }
        terminal_flags = (
            int(termination["tagged"])
            + sum(termination["fallen"])
            + sum(termination["out_of_bounds"])
            + int(termination["timed_out"])
        )
        tag_contact_consistent = bool(
            not termination["tagged"]
            or (
                bool(np.asarray(state.diagnostics.tag_contact))
                and int(np.asarray(state.diagnostics.tag_contact_count)) > 0
                and float(np.asarray(state.diagnostics.tag_minimum_distance)) <= 0.0
            )
        )
        rollouts.append(
            {
                "completed": bool(np.asarray(state.done)),
                "controls_finite": controls_finite,
                "finite": finite,
                "maximum_command_magnitude": maximum_command_magnitude,
                "minimum_agent_distance": minimum_agent_distance,
                "minimum_root_height": minimum_root_height,
                "observation_consistent": observation_consistent,
                "seed": rollout_seed,
                "steps_executed": steps_executed,
                "tag_contact_consistent": tag_contact_consistent,
                "terminal_cause_exclusive": terminal_flags == 1,
                "termination": termination,
            }
        )

    checks = {
        "all_controls_finite": all(r["controls_finite"] for r in rollouts),
        "all_observations_consistent": all(
            r["observation_consistent"] for r in rollouts
        ),
        "all_rollouts_completed": all(r["completed"] for r in rollouts),
        "all_states_finite": all(r["finite"] for r in rollouts),
        "deterministic_reset": deterministic_reset,
        "gpu_backend": jax.default_backend() == "gpu",
        "no_falls": not any(any(r["termination"]["fallen"]) for r in rollouts),
        "no_out_of_bounds": not any(
            any(r["termination"]["out_of_bounds"]) for r in rollouts
        ),
        "tag_contacts_consistent": all(r["tag_contact_consistent"] for r in rollouts),
        "terminal_causes_exclusive": all(
            r["terminal_cause_exclusive"] for r in rollouts
        ),
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
        "experiment": "tag_scripted_rollout_audit",
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "outcomes": {
            "falls": sum(any(r["termination"]["fallen"]) for r in rollouts),
            "out_of_bounds": sum(
                any(r["termination"]["out_of_bounds"]) for r in rollouts
            ),
            "tags": sum(r["termination"]["tagged"] for r in rollouts),
            "timeouts": sum(r["termination"]["timed_out"] for r in rollouts),
        },
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "rollouts": rollouts,
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
