"""Evaluate registered pursuer and evader population matchups.

The headless GPU evaluation restores immutable self-play snapshots, runs the
same reset seeds for every matchup, and writes per-rollout terminal causes plus
a generation-indexed win-rate matrix to JSON. Optional generation filters
support targeted terrain evaluation. It does not mutate the policy population.
"""

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
    TagPursuerConfig,
    TagPursuerEnvironment,
)
from motionforge.policies import FrozenLearnedPursuer
from motionforge.self_play import load_policy_snapshots


@dataclass
class Config:
    population_dir: Path = Path("logs/p8/population")
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 1000
    seeds: int = 4
    episode_duration: float = 20.0
    separation: float = 2.0
    arena_half_extent: float = 4.0
    slope_degrees: float = 0.0
    pursuer_generation: int | None = None
    evader_generation: int | None = None
    action_repeat: int = 5
    naconmax: int = 64
    njmax: int = 256
    output: Path = Path("logs/p8/tag_population_matrix_a.json")


def _terminal_cause(termination) -> str:
    if bool(np.asarray(termination.tagged)):
        return "tagged"
    fallen = np.asarray(termination.fallen)
    out_of_bounds = np.asarray(termination.out_of_bounds)
    if bool(fallen[0]):
        return "pursuer_fall"
    if bool(out_of_bounds[0]):
        return "pursuer_out_of_bounds"
    if bool(fallen[1]):
        return "evader_fall"
    if bool(out_of_bounds[1]):
        return "evader_out_of_bounds"
    if bool(np.asarray(termination.timed_out)):
        return "timed_out"
    return "incomplete"


def main(config: Config) -> None:
    if config.seeds <= 0 or config.episode_duration <= 0.0:
        raise ValueError("seeds and episode_duration must be positive")
    if config.pursuer_generation is not None and config.pursuer_generation < 0:
        raise ValueError("pursuer_generation must be nonnegative")
    if config.evader_generation is not None and config.evader_generation < 0:
        raise ValueError("evader_generation must be nonnegative")
    population_dir = config.population_dir.resolve()
    pursuers = load_policy_snapshots(population_dir, role="pursuer")
    evaders = load_policy_snapshots(population_dir, role="evader")
    if config.pursuer_generation is not None:
        pursuers = [
            item
            for item in pursuers
            if item.generation == config.pursuer_generation
        ]
    if config.evader_generation is not None:
        evaders = [
            item
            for item in evaders
            if item.generation == config.evader_generation
        ]
    if not pursuers or not evaders:
        raise ValueError("generation filters must match pursuer and evader snapshots")

    maximum_steps = round(
        config.episode_duration / (0.02 * config.action_repeat)
    )
    action_scale = jp.asarray([1.0, 0.5, 1.0])
    matchups = []
    all_finite = True
    all_completed = True
    identities_verified = True

    for evader in evaders:
        environment = TagPursuerEnvironment(
            locomotion_checkpoint=config.locomotion_checkpoint,
            evader_checkpoint=population_dir / evader.checkpoint,
            evader_fixed_noise_std=evader.fixed_noise_std,
            tag_config=TagEnvironmentConfig(
                episode_duration=config.episode_duration,
                separation=config.separation,
                arena_half_extent=config.arena_half_extent,
                slope_degrees=config.slope_degrees,
                naconmax=config.naconmax,
                njmax=config.njmax,
            ),
            pursuer_config=TagPursuerConfig(action_repeat=config.action_repeat),
        )
        reset = jax.jit(environment.reset)
        identities_verified &= (
            environment.evader.specification()["checkpoint_digest"]
            == evader.checkpoint_digest
        )

        for pursuer in pursuers:
            policy = FrozenLearnedPursuer(
                agent_index=environment.pursuer_index,
                checkpoint_path=population_dir / pursuer.checkpoint,
                fixed_noise_std=pursuer.fixed_noise_std,
            )
            identities_verified &= (
                policy.specification()["checkpoint_digest"]
                == pursuer.checkpoint_digest
            )

            @jax.jit
            def policy_step(state):
                normalized_action = policy.command(state.obs) / action_scale
                return environment.step(state, normalized_action)

            rollouts = []
            for offset in range(config.seeds):
                rollout_seed = config.seed + offset
                state = reset(jax.random.PRNGKey(rollout_seed))
                finite = True
                steps = 0
                for _ in range(maximum_steps):
                    state = policy_step(state)
                    state.pipeline_state.tag_state.data.qpos.block_until_ready()
                    steps += 1
                    finite &= bool(
                        np.isfinite(
                            np.asarray(state.pipeline_state.tag_state.data.qpos)
                        ).all()
                        and np.isfinite(np.asarray(state.obs)).all()
                    )
                    if bool(np.asarray(state.done)):
                        break
                termination = state.pipeline_state.tag_state.termination
                cause = _terminal_cause(termination)
                rollouts.append(
                    {
                        "completed": bool(np.asarray(state.done)),
                        "finite": finite,
                        "seed": rollout_seed,
                        "steps": steps,
                        "terminal_cause": cause,
                    }
                )

            counts = {
                cause: sum(item["terminal_cause"] == cause for item in rollouts)
                for cause in (
                    "tagged",
                    "pursuer_fall",
                    "pursuer_out_of_bounds",
                    "evader_fall",
                    "evader_out_of_bounds",
                    "timed_out",
                    "incomplete",
                )
            }
            pursuer_wins = counts["tagged"] + counts["evader_fall"] + counts[
                "evader_out_of_bounds"
            ]
            evader_wins = counts["timed_out"] + counts["pursuer_fall"] + counts[
                "pursuer_out_of_bounds"
            ]
            all_finite &= all(item["finite"] for item in rollouts)
            all_completed &= all(item["completed"] for item in rollouts)
            matchups.append(
                {
                    "cause_counts": counts,
                    "evader_generation": evader.generation,
                    "evader_snapshot_id": evader.snapshot_id,
                    "evader_win_rate": evader_wins / config.seeds,
                    "pursuer_generation": pursuer.generation,
                    "pursuer_snapshot_id": pursuer.snapshot_id,
                    "pursuer_win_rate": pursuer_wins / config.seeds,
                    "rollouts": rollouts,
                }
            )

    checks = {
        "all_matchups_completed": all_completed,
        "all_rollouts_finite": all_finite,
        "gpu_backend": jax.default_backend() == "gpu",
        "matchup_count": len(matchups) == len(pursuers) * len(evaders),
        "snapshot_identities_verified": identities_verified,
        "outcomes_accounted_for": all(
            sum(matchup["cause_counts"].values()) == config.seeds
            for matchup in matchups
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "checks": checks,
        "config": {
            **asdict(config),
            "locomotion_checkpoint": str(config.locomotion_checkpoint),
            "output": str(config.output),
            "population_dir": str(config.population_dir),
        },
        "evader_generations": [item.generation for item in evaders],
        "experiment": "p8_tag_population_evaluation",
        "matchups": matchups,
        "passed": all(checks.values()),
        "pursuer_generations": [item.generation for item in pursuers],
        "python_version": platform.python_version(),
        "seed": config.seed,
        "seeds": config.seeds,
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
