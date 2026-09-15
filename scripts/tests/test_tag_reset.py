"""Validate seeded, symmetric, face-to-face two-G1 reset sampling."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import (
    TagResetConfig,
    build_tag_reset_layout,
    build_two_g1_model,
    make_two_g1_data,
    sample_tag_reset,
)


@dataclass
class Config:
    seed: int = 0
    alternate_seed: int = 1
    separation: float = 2.0
    arena_half_extent: float = 4.0
    output: Path = Path("logs/p3/tag_reset_a.json")


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main(config: Config) -> None:
    if config.alternate_seed == config.seed:
        raise ValueError("alternate_seed must differ from seed")
    if config.arena_half_extent <= 0.0:
        raise ValueError("arena_half_extent must be positive")

    bundle = build_two_g1_model()
    template_data = make_two_g1_data(bundle, separation=config.separation)
    layout = build_tag_reset_layout(bundle)
    reset_config = TagResetConfig(separation=config.separation)

    template_qpos = jax.numpy.asarray(template_data.qpos)
    template_qvel = jax.numpy.asarray(template_data.qvel)

    @jax.jit
    def reset(key):
        return sample_tag_reset(
            template_qpos,
            template_qvel,
            key,
            layout,
            reset_config,
        )

    first = reset(jax.random.PRNGKey(config.seed))
    repeated = reset(jax.random.PRNGKey(config.seed))
    alternate = reset(jax.random.PRNGKey(config.alternate_seed))
    first.qpos.block_until_ready()

    first_qpos = np.asarray(first.qpos)
    first_qvel = np.asarray(first.qvel)
    repeated_qpos = np.asarray(repeated.qpos)
    alternate_qpos = np.asarray(alternate.qpos)
    positions = np.asarray(first.planar_positions)
    headings = np.asarray(first.headings)

    separation = float(np.linalg.norm(positions[1] - positions[0]))
    center = np.mean(positions, axis=0)
    quaternion_norms = np.asarray(
        [
            np.linalg.norm(
                first_qpos[
                    agent.root_qpos_start + 3 : agent.root_qpos_start + 7
                ]
            )
            for agent in layout.agents
        ]
    )

    heading_vectors = np.stack([np.cos(headings), np.sin(headings)], axis=-1)
    directions_to_opponent = np.stack(
        [positions[1] - positions[0], positions[0] - positions[1]]
    )
    directions_to_opponent /= np.linalg.norm(
        directions_to_opponent,
        axis=-1,
        keepdims=True,
    )
    facing_alignment = np.sum(
        heading_vectors * directions_to_opponent,
        axis=-1,
    )

    joint_pose_preserved = all(
        np.allclose(
            first_qpos[agent.joint_qpos_slice],
            np.asarray(template_data.qpos)[agent.joint_qpos_slice],
            atol=1e-7,
            rtol=0.0,
        )
        for agent in layout.agents
    )

    checks = {
        "alternate_seed_changes_spawn": bool(
            not np.array_equal(first_qpos, alternate_qpos)
        ),
        "agents_face_each_other": bool(np.all(facing_alignment > 0.99999)),
        "agents_within_bounds": bool(
            np.all(np.abs(positions) <= config.arena_half_extent)
        ),
        "centered_spawn": bool(np.allclose(center, 0.0, atol=1e-6)),
        "finite_state": bool(
            np.isfinite(first_qpos).all() and np.isfinite(first_qvel).all()
        ),
        "joint_pose_preserved": bool(joint_pose_preserved),
        "quaternions_normalized": bool(
            np.allclose(quaternion_norms, 1.0, atol=1e-6)
        ),
        "same_seed_exact": bool(np.array_equal(first_qpos, repeated_qpos)),
        "separation_exact": bool(
            np.isclose(separation, config.separation, atol=1e-6)
        ),
        "velocities_zero": bool(np.array_equal(first_qvel, np.zeros_like(first_qvel))),
    }

    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_deterministic_reset",
        "facing_alignment": facing_alignment.tolist(),
        "headings": headings.tolist(),
        "jax_version": jax.__version__,
        "planar_positions": positions.tolist(),
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "qpos_sha256": array_digest(first_qpos),
        "qvel_sha256": array_digest(first_qvel),
        "quaternion_norms": quaternion_norms.tolist(),
        "seed": config.seed,
        "separation": separation,
        "spawn_angle": float(np.asarray(first.spawn_angle)),
        "mujoco_version": version("mujoco"),
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
