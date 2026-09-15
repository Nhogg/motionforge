"""Validate deterministic episode-timeout boundary semantics."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import (
    EpisodeTimeoutConfig,
    advance_episode_timeout,
    observe_episode_timeout,
)


@dataclass
class Config:
    seed: int = 0
    duration: float = 20.0
    control_timestep: float = 0.02
    output: Path = Path("logs/p3/tag_timeout_a.json")


def observation_result(observation) -> dict[str, object]:
    return {
        "elapsed_seconds": float(np.asarray(observation.elapsed_seconds)),
        "step_count": int(np.asarray(observation.step_count)),
        "steps_remaining": int(np.asarray(observation.steps_remaining)),
        "timed_out": bool(np.asarray(observation.timed_out)),
    }


def main(config: Config) -> None:
    timeout_config = EpisodeTimeoutConfig(
        duration=config.duration,
        control_timestep=config.control_timestep,
    )
    maximum_steps = timeout_config.maximum_steps

    observe = jax.jit(
        lambda step: observe_episode_timeout(step, timeout_config)
    )
    advance = jax.jit(
        lambda step: advance_episode_timeout(step, timeout_config)
    )

    initial = observe(0)
    penultimate = observe(maximum_steps - 1)
    exact = observe(maximum_steps)
    beyond = observe(maximum_steps + 1)
    advanced_to_timeout = advance(maximum_steps - 1)
    exact.elapsed_seconds.block_until_ready()

    checks = {
        "advance_hits_exact_boundary": bool(
            np.asarray(advanced_to_timeout.timed_out)
            and int(np.asarray(advanced_to_timeout.step_count)) == maximum_steps
        ),
        "beyond_boundary_remains_timed_out": bool(
            np.asarray(beyond.timed_out)
        ),
        "elapsed_time_matches_duration": bool(
            np.isclose(
                float(np.asarray(exact.elapsed_seconds)),
                config.duration,
                atol=1e-6,
            )
        ),
        "exact_boundary_times_out": bool(np.asarray(exact.timed_out)),
        "initial_not_timed_out": bool(not np.asarray(initial.timed_out)),
        "initial_steps_remaining": bool(
            int(np.asarray(initial.steps_remaining)) == maximum_steps
        ),
        "penultimate_not_timed_out": bool(
            not np.asarray(penultimate.timed_out)
        ),
        "penultimate_step_remaining": bool(
            int(np.asarray(penultimate.steps_remaining)) == 1
        ),
        "remaining_steps_clamped": bool(
            int(np.asarray(beyond.steps_remaining)) == 0
        ),
    }

    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_episode_timeout",
        "fixtures": {
            "advanced_to_timeout": observation_result(advanced_to_timeout),
            "beyond": observation_result(beyond),
            "exact": observation_result(exact),
            "initial": observation_result(initial),
            "penultimate": observation_result(penultimate),
        },
        "jax_version": jax.__version__,
        "maximum_steps": maximum_steps,
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
