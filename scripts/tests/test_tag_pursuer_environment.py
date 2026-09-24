"""Audit the P7 single-agent pursuer environment contract on MJX-Warp."""

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
    TagPursuerConfig,
    TagPursuerEnvironment,
    pursuer_action_to_command,
    pursuer_reward,
    wrap_tag_pursuer_for_training,
)


@dataclass
class Config:
    checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    action_repeat: int = 5
    output: Path = Path("logs/p7/tag_pursuer_environment_a.json")


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
    pursuer_config = TagPursuerConfig(action_repeat=config.action_repeat)
    environment = TagPursuerEnvironment(
        locomotion_checkpoint=config.checkpoint,
        tag_config=tag_config,
        pursuer_config=pursuer_config,
    )
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)

    state = reset(jax.random.PRNGKey(config.seed))
    action = jp.asarray([1.5, -2.0, 0.25])
    expected_command = jp.asarray([1.0, -0.5, 0.25])
    next_state = step(state, action)
    next_state.pipeline_state.tag_state.data.qpos.block_until_ready()

    timeout_tag_state = state.pipeline_state.tag_state.replace(
        step_count=jp.asarray(
            environment.tag_environment.timeout_config.maximum_steps - 1,
            dtype=jp.int32,
        )
    )
    timeout_state = state.replace(
        pipeline_state=state.pipeline_state.replace(tag_state=timeout_tag_state)
    )
    timed_out = step(timeout_state, jp.zeros(3))
    timed_out.pipeline_state.tag_state.data.qpos.block_until_ready()

    wrapped = wrap_tag_pursuer_for_training(
        environment,
        episode_length=environment.tag_environment.timeout_config.maximum_steps,
    )
    batch_keys = jax.random.split(jax.random.PRNGKey(config.seed + 1), 2)
    wrapped_state = jax.jit(wrapped.reset)(batch_keys)
    near_timeout_tag_state = wrapped_state.pipeline_state.tag_state.replace(
        step_count=jp.full(
            (2,),
            environment.tag_environment.timeout_config.maximum_steps - 1,
            dtype=jp.int32,
        )
    )
    wrapped_state = wrapped_state.replace(
        pipeline_state=wrapped_state.pipeline_state.replace(
            tag_state=near_timeout_tag_state
        )
    )
    wrapped_next = jax.jit(wrapped.step)(wrapped_state, jp.zeros((2, 3)))
    wrapped_next.pipeline_state.tag_state.data.qpos.block_until_ready()

    zero_command = jp.zeros(3)
    tag_terms = pursuer_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero_command,
        current_command=zero_command,
        termination=termination(tagged=True),
        pursuer_index=0,
        config=pursuer_config,
    )
    fall_terms = pursuer_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero_command,
        current_command=zero_command,
        termination=termination(fallen=(True, False)),
        pursuer_index=0,
        config=pursuer_config,
    )
    bounds_terms = pursuer_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero_command,
        current_command=zero_command,
        termination=termination(out_of_bounds=(True, False)),
        pursuer_index=0,
        config=pursuer_config,
    )
    evader_failure_terms = pursuer_reward(
        previous_distance=jp.asarray(1.0),
        current_distance=jp.asarray(1.0),
        previous_command=zero_command,
        current_command=zero_command,
        termination=termination(fallen=(False, True)),
        pursuer_index=0,
        config=pursuer_config,
    )
    progress_terms = pursuer_reward(
        previous_distance=jp.asarray(2.0),
        current_distance=jp.asarray(1.5),
        previous_command=zero_command,
        current_command=jp.ones(3),
        termination=termination(),
        pursuer_index=0,
        config=pursuer_config,
    )

    initial_obs = np.asarray(state.obs)
    next_obs = np.asarray(next_state.obs)
    metric_values = np.asarray(list(next_state.metrics.values()))
    checks = {
        "action_clipped_and_scaled": bool(
            np.allclose(np.asarray(pursuer_action_to_command(action)), expected_command)
        ),
        "action_size": environment.action_size == 3,
        "backend_gpu": jax.default_backend() == "gpu",
        "command_held_for_action_repeat": (
            int(np.asarray(next_state.pipeline_state.tag_state.step_count))
            == config.action_repeat
        ),
        "evader_failure_not_rewarded": bool(
            np.isclose(np.asarray(evader_failure_terms.pursuer_fall), 0.0)
            and np.isclose(np.asarray(evader_failure_terms.tag), 0.0)
        ),
        "finite_step": bool(
            np.isfinite(np.asarray(next_state.pipeline_state.tag_state.data.qpos)).all()
            and np.isfinite(
                np.asarray(next_state.pipeline_state.tag_state.data.qvel)
            ).all()
            and np.isfinite(next_obs).all()
            and np.isfinite(metric_values).all()
            and np.isfinite(np.asarray(next_state.reward))
        ),
        "observation_has_no_clock": initial_obs.shape == (9,),
        "observation_size": environment.observation_size == 9,
        "previous_command_observed": bool(
            np.allclose(next_obs[-3:], np.asarray([1.0, -1.0, 0.25]))
        ),
        "reward_bounds_penalty": bool(
            np.isclose(
                np.asarray(bounds_terms.pursuer_out_of_bounds),
                -pursuer_config.pursuer_out_of_bounds_penalty,
            )
        ),
        "reward_command_change_penalty": bool(
            np.isclose(
                np.asarray(progress_terms.command_change),
                -3.0 * pursuer_config.command_change_penalty_scale,
            )
        ),
        "reward_fall_penalty": bool(
            np.isclose(
                np.asarray(fall_terms.pursuer_fall),
                -pursuer_config.pursuer_fall_penalty,
            )
        ),
        "reward_progress": bool(np.isclose(np.asarray(progress_terms.progress), 1.0)),
        "reward_tag_bonus": bool(
            np.isclose(np.asarray(tag_terms.tag), pursuer_config.tag_reward)
        ),
        "timeout_done": bool(np.asarray(timed_out.done)),
        "timeout_is_truncation": bool(np.asarray(timed_out.info["truncation"])),
        "timeout_remains_sticky_through_repeat": (
            int(np.asarray(timed_out.pipeline_state.tag_state.step_count))
            == environment.tag_environment.timeout_config.maximum_steps
            + config.action_repeat
            - 1
        ),
        "wrapped_autoreset_preserves_done": bool(np.asarray(wrapped_next.done).all()),
        "wrapped_autoreset_resets_pipeline": bool(
            (np.asarray(wrapped_next.pipeline_state.tag_state.step_count) == 0).all()
        ),
        "wrapped_timeout_is_truncation": bool(
            np.asarray(wrapped_next.info["truncation"]).all()
            and np.asarray(wrapped_next.info["time_out"]).all()
        ),
        "wrapper_bookkeeping_preserved": all(
            key in wrapped_next.info
            for key in (
                "episode_done",
                "episode_metrics",
                "steps",
                "time_out",
                "truncation",
            )
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "output": str(config.output),
        },
        "evader_fingerprint": environment.evader.fingerprint,
        "experiment": "p7_tag_pursuer_environment",
        "jax_version": jax.__version__,
        "metrics": {
            key: float(np.asarray(value)) for key, value in next_state.metrics.items()
        },
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
