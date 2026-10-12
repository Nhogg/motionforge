"""Evaluate a hierarchical TAG checkpoint deterministically in both roles."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.league import FixedLeague, LeagueOpponent
from motionforge.models import (
    load_p2_checkpoint,
    migrate_p2_actor_parameters,
    p2_locomotion_normalization,
)
from motionforge.rollout import (
    HierarchicalRolloutConfig,
    HierarchicalRolloutRunner,
    collect_hierarchical_rollout,
)
from motionforge.training import (
    HierarchicalPpoUpdater,
    HierarchicalUpdateConfig,
    load_hierarchical_checkpoint,
    read_hierarchical_checkpoint_manifest,
)


@dataclass
class Config:
    checkpoint: Path = Path("???")
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    seed: int = 0
    seeds: int = 4
    duration: float = 10.0
    naconmax: int = 256
    njmax: int = 512
    output: Path = Path("logs/hierarchical/evaluation/tag_checkpoint_a.json")


def _validate_config(config: Config) -> None:
    if config.seeds <= 0:
        raise ValueError("seeds must be positive")
    if config.duration <= 0.0:
        raise ValueError("duration must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("naconmax and njmax must be positive")
    if not config.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {config.checkpoint}")
    if not config.locomotion_checkpoint.is_dir():
        raise FileNotFoundError(
            f"locomotion checkpoint does not exist: {config.locomotion_checkpoint}"
        )


def _rollout_metrics(batch, learner_index: int, role: str) -> dict[str, object]:
    valid = np.asarray(batch.transition_valid, dtype=bool)
    transitions = batch.transitions
    fallen = np.asarray(transitions.fallen)[:, learner_index]
    out_of_bounds = np.asarray(transitions.out_of_bounds)[:, learner_index]
    tagged = np.asarray(transitions.tagged, dtype=bool)
    distances = np.linalg.norm(
        np.asarray(transitions.relative_position)[:, learner_index], axis=-1
    )
    tracking_error = np.asarray(transitions.command_tracking_error)
    up_alignment = np.clip(
        np.asarray(transitions.up_alignment)[:, learner_index], -1.0, 1.0
    )
    physical_survival = not bool(np.any((fallen | out_of_bounds) & valid))
    tag_occurred = bool(np.any(tagged & valid))
    role_success = (
        tag_occurred if role == "pursuer" else physical_survival and not tag_occurred
    )
    return {
        "command_tracking_rmse": float(
            np.sqrt(np.mean(np.square(tracking_error[valid])))
        ),
        "final_distance": float(distances[np.flatnonzero(valid)[-1]]),
        "locomotion_reward": float(
            np.asarray(batch.locomotion.reward)[
                np.asarray(batch.locomotion.valid)
            ].sum()
        ),
        "maximum_command_norm": float(
            np.linalg.norm(np.asarray(batch.strategy.command), axis=-1).max()
        ),
        "maximum_tilt_degrees": float(np.degrees(np.arccos(up_alignment[valid])).max()),
        "minimum_distance": float(distances[valid].min()),
        "minimum_root_height": float(
            np.asarray(transitions.root_height)[:, learner_index][valid].min()
        ),
        "physical_survival": physical_survival,
        "role_success": role_success,
        "steps_alive": int(valid.sum()),
        "strategy_reward": float(
            np.asarray(batch.strategy.reward)[np.asarray(batch.strategy.valid)].sum()
        ),
        "tag_occurred": tag_occurred,
    }


def main(config: Config) -> None:
    _validate_config(config)
    checkpoint_manifest = read_hierarchical_checkpoint_manifest(config.checkpoint)
    update_config = HierarchicalUpdateConfig(**checkpoint_manifest["update_config"])
    _, p2_parameters = load_p2_checkpoint(config.locomotion_checkpoint)
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None)),
        config=MultiAgentTaskConfig(
            naconmax=config.naconmax,
            njmax=config.njmax,
        ),
    )
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2_parameters[0]),
        HierarchicalRolloutConfig(stochastic_actions=False),
    )
    initial = runner.reset(jax.random.PRNGKey(config.seed))
    warm_start_parameters = migrate_p2_actor_parameters(
        runner.initialize_parameters(jax.random.PRNGKey(config.seed + 1), initial),
        p2_parameters[1],
    )
    updater = HierarchicalPpoUpdater(update_config)
    template = updater.initialize(warm_start_parameters)
    train_state, _ = load_hierarchical_checkpoint(config.checkpoint, template)
    environment.league = FixedLeague(
        LeagueOpponent(
            "warm-start/p2",
            "warm_start",
            warm_start_parameters,
        )
    )
    horizon = round(config.duration / environment.config.control_timestep)
    if horizon % runner.config.locomotion_steps_per_strategy_step != 0:
        raise ValueError(
            "duration must produce a horizon divisible by strategy cadence"
        )

    scenarios = []
    all_finite = True
    all_rollouts_valid = True
    for learner_is_pursuer in (True, False):
        role = "pursuer" if learner_is_pursuer else "evader"
        rollouts = []
        for rollout_seed in range(config.seed, config.seed + config.seeds):
            state = runner.reset(jax.random.PRNGKey(rollout_seed))
            segment = runner.start_segment(
                train_state.parameters,
                seed=rollout_seed,
                learner_is_pursuer=learner_is_pursuer,
            )
            _, batch = collect_hierarchical_rollout(
                runner,
                segment,
                state,
                horizon=horizon,
                episode_id=jp.asarray(rollout_seed),
                environment_id=jp.asarray(0),
            )
            batch.transitions.root_height.block_until_ready()
            metrics = _rollout_metrics(batch, segment.learner_index, role)
            metrics["seed"] = rollout_seed
            rollouts.append(metrics)
            all_finite = all_finite and all(
                not isinstance(value, float) or np.isfinite(value)
                for value in metrics.values()
            )
            all_rollouts_valid = all_rollouts_valid and metrics["steps_alive"] > 0
        scenarios.append(
            {
                "mean_command_tracking_rmse": float(
                    np.mean([item["command_tracking_rmse"] for item in rollouts])
                ),
                "mean_final_distance": float(
                    np.mean([item["final_distance"] for item in rollouts])
                ),
                "physical_survival_rate": float(
                    np.mean([item["physical_survival"] for item in rollouts])
                ),
                "role": role,
                "role_success_rate": float(
                    np.mean([item["role_success"] for item in rollouts])
                ),
                "rollouts": rollouts,
                "tag_rate": float(np.mean([item["tag_occurred"] for item in rollouts])),
            }
        )

    checks = {
        "all_rollouts_valid": all_rollouts_valid,
        "finite_metrics": all_finite,
        "gpu_backend": jax.default_backend() == "gpu",
        "parameters_loaded": int(train_state.iteration)
        == checkpoint_manifest["iteration"],
        "role_count": len(scenarios) == 2,
    }
    payload = {
        "backend": jax.default_backend(),
        "checkpoint": str(config.checkpoint.resolve()),
        "checkpoint_iteration": int(train_state.iteration),
        "checks": checks,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "locomotion_checkpoint": str(config.locomotion_checkpoint),
            "output": str(config.output),
        },
        "control_timestep": environment.config.control_timestep,
        "experiment": "hierarchical_tag_checkpoint_evaluation",
        "jax_version": jax.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "rollout_steps": horizon,
        "scenarios": scenarios,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
