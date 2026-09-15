"""Exercise the integrated two-G1 MJX-Warp tag environment.

The test runs deterministic reset and step fixtures for observations and every
terminal cause, then writes a machine-readable JSON result. It requires a JAX
GPU backend with MJX-Warp support.
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
from mujoco import mjx

from motionforge.cli import run_hydra
from motionforge.envs import TagEnvironmentConfig, TwoG1TagEnvironment


@dataclass
class Config:
    seed: int = 0
    separation: float = 2.0
    episode_duration: float = 20.0
    arena_half_extent: float = 4.0
    fall_persistence_steps: int = 5
    naconmax: int = 64
    njmax: int = 256
    output: Path = Path("logs/p3/tag_environment_a.json")


def main(config: Config) -> None:
    environment = TwoG1TagEnvironment(
        TagEnvironmentConfig(
            episode_duration=config.episode_duration,
            separation=config.separation,
            arena_half_extent=config.arena_half_extent,
            fall_persistence_steps=config.fall_persistence_steps,
            naconmax=config.naconmax,
            njmax=config.njmax,
        )
    )
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)

    initial = reset(jax.random.PRNGKey(config.seed))
    targets = environment.default_joint_targets
    normal = step(initial, targets)

    timeout_start = initial.replace(
        step_count=jp.asarray(
            environment.timeout_config.maximum_steps - 1,
            dtype=jp.int32,
        )
    )
    timed_out = step(timeout_start, targets)

    agent0_root = environment.fall_layout.agents[0].root_qpos_start
    fallen_qpos = initial.data.qpos.at[
        agent0_root + 3 : agent0_root + 7
    ].set(jp.asarray([0.0, 1.0, 0.0, 0.0]))
    fallen = initial.replace(
        data=mjx.forward(
            environment.model,
            initial.data.replace(qpos=fallen_qpos),
        ),
        fall_counts=jp.asarray(
            [config.fall_persistence_steps - 1, 0],
            dtype=jp.int32,
        ),
    )
    fallen = step(fallen, targets)

    agent1_root = environment.bounds_layout.agents[1].root_qpos_start
    oob_qpos = initial.data.qpos.at[agent1_root].set(
        config.arena_half_extent + 0.5
    )
    oob = initial.replace(
        data=mjx.forward(
            environment.model,
            initial.data.replace(qpos=oob_qpos),
        )
    )
    oob = step(oob, targets)

    agent0_start = environment.reset_layout.agents[0].root_qpos_start
    agent1_start = environment.reset_layout.agents[1].root_qpos_start
    overlap_qpos = initial.data.qpos.at[
        agent1_start : agent1_start + 3
    ].set(initial.data.qpos[agent0_start : agent0_start + 3])
    overlap = initial.replace(
        data=mjx.forward(
            environment.model,
            initial.data.replace(qpos=overlap_qpos),
        )
    )
    overlap = step(overlap, targets)

    for state in (normal, timed_out, fallen, oob, overlap):
        state.data.qpos.block_until_ready()

    initial_relative_position = np.asarray(
        initial.observation.relative_position
    )
    initial_relative_velocity = np.asarray(
        initial.observation.relative_velocity
    )
    normal_relative_position = np.asarray(normal.observation.relative_position)
    normal_relative_velocity = np.asarray(normal.observation.relative_velocity)

    checks = {
        "backend_gpu": jax.default_backend() == "gpu",
        "fall_cause_isolated": (
            np.asarray(fallen.termination.fallen).tolist() == [True, False]
        ),
        "fall_done": bool(np.asarray(fallen.done)),
        "fall_persistence": (
            int(np.asarray(fallen.fall_counts[0]))
            >= config.fall_persistence_steps
        ),
        "initial_relative_positions": bool(
            np.allclose(
                initial_relative_position,
                np.asarray([[config.separation, 0.0]] * 2),
                atol=1e-5,
            )
        ),
        "initial_relative_velocities": bool(
            np.allclose(initial_relative_velocity, 0.0, atol=1e-6)
        ),
        "normal_not_done": not bool(np.asarray(normal.done)),
        "observation_shape": (
            initial_relative_position.shape == (2, 2)
            and initial_relative_velocity.shape == (2, 2)
            and normal_relative_position.shape == (2, 2)
            and normal_relative_velocity.shape == (2, 2)
        ),
        "observations_finite": bool(
            np.isfinite(normal_relative_position).all()
            and np.isfinite(normal_relative_velocity).all()
        ),
        "oob_cause_isolated": (
            np.asarray(oob.termination.out_of_bounds).tolist()
            == [False, True]
        ),
        "oob_done": bool(np.asarray(oob.done)),
        "reset_causes_clear": bool(
            not np.asarray(initial.termination.tagged)
            and not np.asarray(initial.termination.fallen).any()
            and not np.asarray(initial.termination.out_of_bounds).any()
            and not np.asarray(initial.termination.timed_out)
        ),
        "reset_not_done": not bool(np.asarray(initial.done)),
        "state_finite": bool(
            np.isfinite(np.asarray(normal.data.qpos)).all()
            and np.isfinite(np.asarray(normal.data.qvel)).all()
        ),
        "tag_cause": bool(np.asarray(overlap.termination.tagged)),
        "tag_done": bool(np.asarray(overlap.done)),
        "timeout_cause": bool(np.asarray(timed_out.termination.timed_out)),
        "timeout_done": bool(np.asarray(timed_out.done)),
    }

    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_tag_environment_integration",
        "fall_counts": np.asarray(fallen.fall_counts).tolist(),
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "seed": config.seed,
        "tag_contact_count": int(
            np.asarray(overlap.diagnostics.tag_contact_count)
        ),
        "timeout_step": int(np.asarray(timed_out.step_count)),
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
