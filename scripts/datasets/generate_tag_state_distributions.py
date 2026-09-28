"""Generate equal-budget P7 TAG state datasets for three command sources.

Inputs are frozen low-level, pursuer, and evader checkpoints plus an exact
shared-state sample budget. Outputs are one JSONL dataset per condition and a
machine-readable suite manifest. Every condition uses identical physics,
resets, control rate, and sample semantics; only the 10 Hz command source
changes between learned, scripted, and seeded random commands.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.controllers import (
    G1LocomotionController,
    build_g1_tag_policy_observation_layout,
    g1_tag_policy_observation,
)
from motionforge.datasets import (
    TAG_STATE_DATASET_SCHEMA_VERSION,
    JsonlDatasetWriter,
    extract_tag_state_sample,
    json_default,
    tag_state_record,
)
from motionforge.envs import TagEnvironmentConfig, TwoG1TagEnvironment
from motionforge.envs.g1_standing import G1StandingJoystick, default_config
from motionforge.envs.tag_pursuer import pursuer_observation
from motionforge.policies import (
    FrozenLearnedEvader,
    FrozenLearnedPursuer,
    ScriptedEvaderConfig,
    ScriptedPursuerConfig,
    scripted_evader_command,
    scripted_pursuer_command,
)

CONDITIONS = ("learned_tag", "scripted_tag", "random_commands")


@dataclass
class Config:
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    pursuer_checkpoint: Path = Path(
        "logs/p7/training/tag_pursuer_fixed_noise_1m_seed0/checkpoints/000001146880"
    )
    evader_checkpoint: Path = Path(
        "logs/p7/training/tag_evader_fixed_noise_1m_seed0_retry/checkpoints/000000327680"
    )
    seed: int = 0
    samples_per_condition: int = 1_024
    high_level_action_repeat: int = 5
    episode_duration: float = 20.0
    separation: float = 2.0
    arena_half_extent: float = 4.0
    naconmax: int = 64
    njmax: int = 256
    near_fall_minimum_root_height: float = 0.60
    near_fall_minimum_up_alignment: float = 0.80
    output_dir: Path = Path("logs/p7/tag_state_distributions_a")


def _validate_config(config: Config) -> None:
    if config.samples_per_condition <= 0:
        raise ValueError("samples_per_condition must be positive")
    if config.high_level_action_repeat <= 0:
        raise ValueError("high_level_action_repeat must be positive")
    if config.episode_duration <= 0.0 or config.separation <= 0.0:
        raise ValueError("episode_duration and separation must be positive")
    if config.arena_half_extent <= 0.0:
        raise ValueError("arena_half_extent must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("contact capacities must be positive")
    if config.near_fall_minimum_root_height <= 0.0:
        raise ValueError("near-fall root height must be positive")
    if not 0.0 <= config.near_fall_minimum_up_alignment <= 1.0:
        raise ValueError("near-fall alignment must be in [0, 1]")


class _Summary:
    """Small streaming physical summary; raw records remain authoritative."""

    def __init__(self) -> None:
        self.distance: list[float] = []
        self.planar_speed: list[float] = []
        self.angular_speed: list[float] = []
        self.joint_speed: list[float] = []
        self.root_height: list[float] = []
        self.up_alignment: list[float] = []
        self.near_falls = 0
        self.falls = 0
        self.contacts = 0

    def add(self, record: dict) -> None:
        linear = np.asarray(record["root_generalized_linear_velocity"])
        angular = np.asarray(record["root_generalized_angular_velocity"])
        joint = np.asarray(record["joint_velocity"])
        self.distance.append(record["agent_distance"])
        self.planar_speed.extend(np.linalg.norm(linear[:, :2], axis=1).tolist())
        self.angular_speed.extend(np.linalg.norm(angular, axis=1).tolist())
        self.joint_speed.extend(np.linalg.norm(joint, axis=1).tolist())
        self.root_height.extend(record["root_height"])
        self.up_alignment.extend(record["up_alignment"])
        self.near_falls += sum(record["near_fall"])
        self.falls += sum(record["fallen"])
        self.contacts += int(record["tag_contact"])

    def result(self, samples: int) -> dict:
        def stats(values: list[float]) -> dict:
            array = np.asarray(values, dtype=np.float64)
            return {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "p05": float(np.quantile(array, 0.05)),
                "p50": float(np.quantile(array, 0.50)),
                "p95": float(np.quantile(array, 0.95)),
            }

        return {
            "agent_distance": stats(self.distance),
            "angular_speed": stats(self.angular_speed),
            "fall_agent_fraction": self.falls / (2 * samples),
            "joint_speed_norm": stats(self.joint_speed),
            "near_fall_agent_fraction": self.near_falls / (2 * samples),
            "planar_speed": stats(self.planar_speed),
            "root_height": stats(self.root_height),
            "tag_contact_state_fraction": self.contacts / samples,
            "up_alignment": stats(self.up_alignment),
        }


def main(config: Config) -> None:
    _validate_config(config)
    for checkpoint_path in (
        config.locomotion_checkpoint,
        config.pursuer_checkpoint,
        config.evader_checkpoint,
    ):
        if not checkpoint_path.resolve().is_dir():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    source_config = default_config()
    source_config.impl = "warp"
    source_config.naconmax = 16
    source_config.njmax = 128
    source_config.push_config.enable = False
    source_environment = G1StandingJoystick(config=source_config)
    controller = G1LocomotionController.from_checkpoint(
        config.locomotion_checkpoint, source_environment, deterministic=True
    )
    learned_pursuer = FrozenLearnedPursuer(0, config.pursuer_checkpoint)
    learned_evader = FrozenLearnedEvader(1, config.evader_checkpoint)

    environment = TwoG1TagEnvironment(
        TagEnvironmentConfig(
            episode_duration=config.episode_duration,
            separation=config.separation,
            arena_half_extent=config.arena_half_extent,
            naconmax=config.naconmax,
            njmax=config.njmax,
        )
    )
    layout = build_g1_tag_policy_observation_layout(environment.model_bundle)
    default_pose = jp.asarray(source_environment._default_pose)
    phase_dt = 2.0 * jp.pi * environment.config.control_timestep * 1.375
    pursuer_config = ScriptedPursuerConfig()
    evader_config = ScriptedEvaderConfig(arena_radius=config.arena_half_extent)
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    act = jax.jit(controller.act)
    extract = jax.jit(
        lambda state, commands, actions, targets: extract_tag_state_sample(
            environment.model_bundle,
            state,
            commands,
            actions,
            targets,
            near_fall_minimum_root_height=config.near_fall_minimum_root_height,
            near_fall_minimum_up_alignment=config.near_fall_minimum_up_alignment,
        )
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    condition_results = []
    for condition in CONDITIONS:
        output = config.output_dir / f"{condition}.jsonl"
        rng = jax.random.PRNGKey(config.seed)
        episode = 0
        episode_seed = config.seed
        rng, reset_rng = jax.random.split(rng)
        state = reset(reset_rng)
        actions = jp.zeros((2, 29), dtype=jp.float32)
        phases = jp.asarray([[0.0, jp.pi], [0.0, jp.pi]], dtype=jp.float32)
        commands = jp.zeros((2, 3), dtype=jp.float32)
        summary = _Summary()
        finite = True
        timestep = 0

        with JsonlDatasetWriter(output) as writer:
            while writer.count < config.samples_per_condition:
                if timestep % config.high_level_action_repeat == 0:
                    if condition == "learned_tag":
                        observations = tuple(
                            pursuer_observation(
                                state.observation,
                                agent_index,
                                commands[agent_index],
                                config.arena_half_extent,
                                2.0,
                            )
                            for agent_index in range(2)
                        )
                        commands = jp.stack(
                            [
                                learned_pursuer.command(observations[0]),
                                learned_evader.command(observations[1]),
                            ]
                        )
                    elif condition == "scripted_tag":
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
                    else:
                        rng, command_rng = jax.random.split(rng)
                        commands = jax.random.uniform(
                            command_rng,
                            (2, 3),
                            minval=-jp.asarray([1.0, 0.5, 1.0]),
                            maxval=jp.asarray([1.0, 0.5, 1.0]),
                        )

                low_observations = tuple(
                    g1_tag_policy_observation(
                        state.data,
                        layout,
                        agent_index,
                        commands[agent_index],
                        actions[agent_index],
                        phases[agent_index],
                        default_pose,
                    )
                    for agent_index in range(2)
                )
                rng, agent0_rng, agent1_rng = jax.random.split(rng, 3)
                controls = (
                    act(low_observations[0], agent0_rng),
                    act(low_observations[1], agent1_rng),
                )
                actions = jp.stack([control.normalized_action for control in controls])
                targets = jp.stack(
                    [control.joint_position_targets for control in controls]
                )
                state = step(state, targets)
                phases = jp.fmod(phases + phase_dt + jp.pi, 2.0 * jp.pi) - jp.pi
                timestep += 1
                sample = extract(state, commands, actions, targets)
                record = tag_state_record(
                    sample,
                    dataset_kind=condition,
                    episode=episode,
                    episode_seed=episode_seed,
                    timestep=timestep,
                    state=state,
                )
                numeric = np.concatenate(
                    [
                        np.asarray(record["root_position_world"]).ravel(),
                        np.asarray(record["joint_position"]).ravel(),
                        np.asarray(record["joint_velocity"]).ravel(),
                        np.asarray(record["command_velocity"]).ravel(),
                        np.asarray(record["normalized_action"]).ravel(),
                    ]
                )
                finite &= bool(np.isfinite(numeric).all())
                summary.add(record)
                writer.write(record)

                if bool(np.asarray(state.done)) and writer.count < config.samples_per_condition:
                    episode += 1
                    episode_seed = config.seed + episode
                    rng = jax.random.PRNGKey(episode_seed)
                    rng, reset_rng = jax.random.split(rng)
                    state = reset(reset_rng)
                    actions = jp.zeros((2, 29), dtype=jp.float32)
                    phases = jp.asarray(
                        [[0.0, jp.pi], [0.0, jp.pi]], dtype=jp.float32
                    )
                    commands = jp.zeros((2, 3), dtype=jp.float32)
                    timestep = 0

        condition_results.append(
            {
                "condition": condition,
                "episodes": episode + 1,
                "finite_values": finite,
                "output": str(output),
                "output_sha256": writer.sha256,
                "samples": writer.count,
                "summary": summary.result(writer.count),
            }
        )

    checks = {
        "all_values_finite": all(result["finite_values"] for result in condition_results),
        "equal_sample_budgets": len(
            {result["samples"] for result in condition_results}
        ) == 1,
        "exact_sample_budgets": all(
            result["samples"] == config.samples_per_condition
            for result in condition_results
        ),
        "gpu_backend": jax.default_backend() == "gpu",
        "three_conditions": [result["condition"] for result in condition_results]
        == list(CONDITIONS),
    }
    manifest = {
        "checks": checks,
        "conditions": condition_results,
        "config": asdict(config),
        "control_timestep": environment.config.control_timestep,
        "experiment": "p7_tag_state_distribution_dataset",
        "high_level_timestep": (
            environment.config.control_timestep * config.high_level_action_repeat
        ),
        "jax_version": jax.__version__,
        "locomotion_checkpoint": str(controller.checkpoint_path),
        "mujoco_version": version("mujoco"),
        "passed": bool(all(checks.values())),
        "policy_specifications": {
            "evader": learned_evader.specification(),
            "pursuer": learned_pursuer.specification(),
        },
        "python_version": platform.python_version(),
        "sample_semantics": (
            "post-step shared state with the command, action, and joint target "
            "that produced it"
        ),
        "schema_version": TAG_STATE_DATASET_SCHEMA_VERSION,
        "seed": config.seed,
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, default=json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, default=json_default, sort_keys=True))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
