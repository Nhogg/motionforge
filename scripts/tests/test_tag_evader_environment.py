"""Audit the P7 learned-evader environment and frozen-pursuer contract."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import (
    TagEnvironmentConfig,
    TagEnvironmentTermination,
    TagEvaderConfig,
    TagEvaderEnvironment,
    evader_reward,
    wrap_tag_evader_for_training,
)


@dataclass
class Config:
    pursuer_checkpoint: Path = Path(
        "logs/p7/training/tag_pursuer_fixed_noise_1m_seed0/checkpoints/"
        "000001146880"
    )
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/"
        "000040632320"
    )
    seed: int = 0
    output: Path = Path("logs/p7/tag_evader_environment_a.json")


def termination(
    *,
    tagged: bool = False,
    fallen: tuple[bool, bool] = (False, False),
    out_of_bounds: tuple[bool, bool] = (False, False),
    timed_out: bool = False,
) -> TagEnvironmentTermination:
    return TagEnvironmentTermination(
        tagged=jp.asarray(tagged),
        fallen=jp.asarray(fallen),
        out_of_bounds=jp.asarray(out_of_bounds),
        timed_out=jp.asarray(timed_out),
    )


def main(config: Config) -> None:
    tag_config = TagEnvironmentConfig(pursuer_index=0)
    evader_config = TagEvaderConfig()
    environment = TagEvaderEnvironment(
        locomotion_checkpoint=config.locomotion_checkpoint,
        pursuer_checkpoint=config.pursuer_checkpoint,
        tag_config=tag_config,
        evader_config=evader_config,
    )
    state = jax.jit(environment.reset)(jax.random.PRNGKey(config.seed))
    next_state = jax.jit(environment.step)(state, jp.asarray([0.3, -0.4, 0.2]))
    next_state.pipeline_state.tag_state.data.qpos.block_until_ready()

    wrapped = wrap_tag_evader_for_training(
        environment,
        episode_length=environment.tag_environment.timeout_config.maximum_steps,
    )
    batch_keys = jax.random.split(jax.random.PRNGKey(config.seed + 1), 2)
    wrapped_state = jax.jit(wrapped.reset)(batch_keys)
    wrapped_next = jax.jit(wrapped.step)(wrapped_state, jp.zeros((2, 3)))
    wrapped_next.pipeline_state.tag_state.data.qpos.block_until_ready()

    zero = jp.zeros(3)
    timeout_terms = evader_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero,
        current_command=zero,
        evader_planar_position=zero[:2],
        evader_planar_velocity=zero[:2],
        arena_half_extent=tag_config.arena_half_extent,
        termination=termination(timed_out=True),
        evader_index=environment.evader_index,
        config=evader_config,
    )
    failure_terms = evader_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.5),
        previous_command=zero,
        current_command=zero,
        evader_planar_position=jp.asarray([3.5, 0.0]),
        evader_planar_velocity=zero[:2],
        arena_half_extent=tag_config.arena_half_extent,
        termination=termination(
            tagged=True,
            fallen=(False, True),
            out_of_bounds=(False, True),
        ),
        evader_index=environment.evader_index,
        config=evader_config,
    )
    tag_terms = evader_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero,
        current_command=zero,
        evader_planar_position=zero[:2],
        evader_planar_velocity=zero[:2],
        arena_half_extent=tag_config.arena_half_extent,
        termination=termination(tagged=True),
        evader_index=environment.evader_index,
        config=evader_config,
    )
    fall_terms = evader_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero,
        current_command=zero,
        evader_planar_position=zero[:2],
        evader_planar_velocity=zero[:2],
        arena_half_extent=tag_config.arena_half_extent,
        termination=termination(fallen=(False, True)),
        evader_index=environment.evader_index,
        config=evader_config,
    )
    out_of_bounds_terms = evader_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero,
        current_command=zero,
        evader_planar_position=zero[:2],
        evader_planar_velocity=zero[:2],
        arena_half_extent=tag_config.arena_half_extent,
        termination=termination(out_of_bounds=(False, True)),
        evader_index=environment.evader_index,
        config=evader_config,
    )
    outward_terms = evader_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero,
        current_command=zero,
        evader_planar_position=jp.asarray([3.5, 0.0]),
        evader_planar_velocity=jp.asarray([1.0, 0.0]),
        arena_half_extent=tag_config.arena_half_extent,
        termination=termination(),
        evader_index=environment.evader_index,
        config=evader_config,
    )
    inward_terms = evader_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero,
        current_command=zero,
        evader_planar_position=jp.asarray([3.5, 0.0]),
        evader_planar_velocity=jp.asarray([-1.0, 0.0]),
        arena_half_extent=tag_config.arena_half_extent,
        termination=termination(),
        evader_index=environment.evader_index,
        config=evader_config,
    )

    next_obs = np.asarray(next_state.obs)
    checks = {
        "action_size": environment.action_size == 3,
        "backend_gpu": jax.default_backend() == "gpu",
        "evader_role_bound": environment.evader_index == 1,
        "finite_step": bool(
            np.isfinite(np.asarray(next_state.pipeline_state.tag_state.data.qpos)).all()
            and np.isfinite(next_obs).all()
            and np.isfinite(np.asarray(next_state.reward))
        ),
        "frozen_pursuer_command_recorded": bool(
            np.isfinite(
                np.asarray(next_state.pipeline_state.pursuer_previous_command)
            ).all()
            and not np.allclose(
                np.asarray(next_state.pipeline_state.pursuer_previous_command), 0.0
            )
        ),
        "observation_size": next_obs.shape == (9,),
        "previous_evader_command_observed": bool(
            np.allclose(next_obs[-3:], np.asarray([0.3, -0.4, 0.2]))
        ),
        "reward_boundary_penalty": bool(
            np.isclose(np.asarray(failure_terms.boundary), -0.25)
        ),
        "reward_boundary_outward_velocity": bool(
            np.isclose(
                np.asarray(outward_terms.boundary_outward_velocity),
                -0.5
                * evader_config.boundary_outward_velocity_penalty_scale,
            )
            and np.isclose(
                np.asarray(inward_terms.boundary_outward_velocity),
                0.0,
            )
        ),
        "reward_evader_failure_penalties": bool(
            np.isclose(np.asarray(failure_terms.tag), -evader_config.tag_penalty)
            and np.isclose(
                np.asarray(failure_terms.evader_fall),
                -evader_config.evader_fall_penalty,
            )
            and np.isclose(
                np.asarray(failure_terms.evader_out_of_bounds),
                -evader_config.evader_out_of_bounds_penalty,
            )
        ),
        "reward_failures_worse_than_tag": bool(
            np.asarray(fall_terms.total) < np.asarray(tag_terms.total)
            and np.asarray(out_of_bounds_terms.total)
            < np.asarray(tag_terms.total)
        ),
        "reward_separation": bool(
            np.isclose(np.asarray(failure_terms.separation), 1.0)
        ),
        "reward_timeout_success": bool(
            np.isclose(np.asarray(timeout_terms.timeout), evader_config.timeout_reward)
        ),
        "step_count_matches_repeat": int(
            np.asarray(next_state.pipeline_state.tag_state.step_count)
        )
        == evader_config.action_repeat,
        "wrapped_finite": bool(
            np.isfinite(
                np.asarray(wrapped_next.pipeline_state.tag_state.data.qpos)
            ).all()
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "pursuer_checkpoint": str(config.pursuer_checkpoint),
            "locomotion_checkpoint": str(config.locomotion_checkpoint),
            "output": str(config.output),
        },
        "experiment": "p7_tag_evader_environment",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "pursuer_fingerprint": environment.pursuer.fingerprint,
        "python_version": platform.python_version(),
        "seed": config.seed,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
