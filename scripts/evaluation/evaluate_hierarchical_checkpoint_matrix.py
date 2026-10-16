"""Rank sequential hierarchical TAG checkpoints with matched pairwise games."""

from __future__ import annotations

import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.envs.tag_endings import TagEndCause
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
from motionforge.rollout.tag_reward import TagRewardConfig, tag_reward_config_from_dict
from motionforge.training import (
    HierarchicalPpoUpdater,
    HierarchicalUpdateConfig,
    TagCheckpointElo,
    load_hierarchical_checkpoint,
    read_hierarchical_checkpoint_manifest,
)


@dataclass
class Config:
    run_directory: Path = Path("???")
    action_mode: str = "deterministic"
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_rebuilt_full_seed0/checkpoints/000200048640"
    )
    initial_separation: float = 1.5
    episode_duration: float = 10.0
    duration: float = 10.0
    seed: int = 24_000
    seeds: int = 16
    naconmax: int = 256
    njmax: int = 4_096
    output: Path | None = None


def _validate_config(config: Config) -> None:
    if config.action_mode not in {"deterministic", "stochastic"}:
        raise ValueError("action_mode must be deterministic or stochastic")
    if not config.run_directory.is_dir():
        raise FileNotFoundError(
            f"training run directory does not exist: {config.run_directory}"
        )
    if not config.locomotion_checkpoint.is_dir():
        raise FileNotFoundError(
            f"locomotion checkpoint does not exist: {config.locomotion_checkpoint}"
        )
    if config.initial_separation <= 0.0:
        raise ValueError("initial_separation must be positive")
    if config.episode_duration <= 0.0 or config.duration <= 0.0:
        raise ValueError("episode_duration and duration must be positive")
    if not math.isclose(config.episode_duration, config.duration, abs_tol=1e-9):
        raise ValueError("duration must equal episode_duration for complete games")
    if config.seeds <= 0:
        raise ValueError("seeds must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("naconmax and njmax must be positive")


def _checkpoint_paths(run_directory: Path) -> list[Path]:
    checkpoint_directory = run_directory / "checkpoints"
    if not checkpoint_directory.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {checkpoint_directory}"
        )
    checkpoints = sorted(
        (
            path
            for path in checkpoint_directory.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    if len(checkpoints) < 2:
        raise ValueError(
            "checkpoint matrix evaluation requires at least two checkpoints"
        )
    return checkpoints


def _load_parameters(checkpoint: Path, warm_start_parameters):
    manifest = read_hierarchical_checkpoint_manifest(checkpoint)
    updater = HierarchicalPpoUpdater(
        HierarchicalUpdateConfig(**manifest["update_config"])
    )
    state, _ = load_hierarchical_checkpoint(
        checkpoint,
        updater.initialize(warm_start_parameters),
    )
    return state.parameters, manifest


def _game_metrics(
    batch,
    learner_index: int,
    *,
    control_timestep: float,
) -> dict[str, object]:
    valid = np.asarray(batch.transition_valid, dtype=bool)
    if not np.any(valid):
        raise ValueError("evaluation rollout contains no valid transitions")
    transitions = batch.transitions
    indices = np.flatnonzero(valid)
    final_index = indices[-1]
    cause = TagEndCause(int(np.asarray(transitions.terminal_cause)[final_index]))
    fallen = np.asarray(transitions.persistent_fallen)
    out_of_bounds = np.asarray(transitions.out_of_bounds)
    tagged = bool(np.any(np.asarray(transitions.tagged, dtype=bool) & valid))
    learner_physical_survival = not bool(
        np.any(
            (fallen[:, learner_index] | out_of_bounds[:, learner_index]) & valid
        )
    )
    physical_survival = not bool(
        np.any((fallen | out_of_bounds) & valid[:, None])
    )
    distances = np.asarray(transitions.planar_distance)
    return {
        "terminal_cause": cause.name,
        "valid_tag": cause == TagEndCause.TAG,
        "tagged": tagged,
        "physical_survival": physical_survival,
        "learner_physical_survival": learner_physical_survival,
        "steps_alive": int(valid.sum()),
        "time_to_tag": (
            float(
                (np.flatnonzero(np.asarray(transitions.tagged) & valid)[0] + 1)
                * control_timestep
            )
            if cause == TagEndCause.TAG
            else None
        ),
        "minimum_distance": float(distances[valid].min()),
        "final_distance": float(distances[final_index]),
    }


def _run_game(
    *,
    runner: HierarchicalRolloutRunner,
    environment: MultiAgentTaskEnv,
    learner_parameters,
    opponent_parameters,
    opponent_id: str,
    learner_is_pursuer: bool,
    seed: int,
    strategy_discount: float,
) -> dict[str, object]:
    environment.league = FixedLeague(
        LeagueOpponent(opponent_id, "historical", opponent_parameters)
    )
    state = runner.reset(jax.random.PRNGKey(seed))
    segment = runner.start_segment(
        learner_parameters,
        seed=seed,
        learner_is_pursuer=learner_is_pursuer,
    )
    horizon = round(
        environment.config.episode_duration / environment.config.control_timestep
    )
    _, batch = collect_hierarchical_rollout(
        runner,
        segment,
        state,
        horizon=horizon,
        episode_id=jp.asarray(seed),
        environment_id=jp.asarray(0),
        strategy_discount=strategy_discount,
    )
    batch.transitions.root_height.block_until_ready()
    metrics = _game_metrics(
        batch,
        segment.learner_index,
        control_timestep=environment.config.control_timestep,
    )
    metrics["seed"] = seed
    metrics["learner_role"] = "pursuer" if learner_is_pursuer else "evader"
    return metrics


def _evaluate_pair(
    *,
    current_id: str,
    current_parameters,
    opponent_id: str,
    opponent_parameters,
    runner: HierarchicalRolloutRunner,
    environment: MultiAgentTaskEnv,
    config: Config,
    strategy_discount: float,
    ratings: TagCheckpointElo,
) -> dict[str, object]:
    games = []
    for learner_is_pursuer in (True, False):
        for seed in range(config.seed, config.seed + config.seeds):
            result = _run_game(
                runner=runner,
                environment=environment,
                learner_parameters=current_parameters,
                opponent_parameters=opponent_parameters,
                opponent_id=opponent_id,
                learner_is_pursuer=learner_is_pursuer,
                seed=seed,
                strategy_discount=strategy_discount,
            )
            result["current_checkpoint"] = current_id
            result["opponent_checkpoint"] = opponent_id
            if result["valid_tag"]:
                if learner_is_pursuer:
                    ratings.update(
                        pursuer_id=current_id,
                        evader_id=opponent_id,
                        pursuer_won=True,
                    )
                    result["winner"] = current_id
                else:
                    ratings.update(
                        pursuer_id=opponent_id,
                        evader_id=current_id,
                        pursuer_won=True,
                    )
                    result["winner"] = opponent_id
            else:
                result["winner"] = None
            games.append(result)

    valid_games = [game for game in games if game["valid_tag"]]
    return {
        "current_checkpoint": current_id,
        "opponent_checkpoint": opponent_id,
        "games": games,
        "valid_tag_games": len(valid_games),
        "censored_games": len(games) - len(valid_games),
        "physical_failure_games": sum(
            not bool(game["physical_survival"]) for game in games
        ),
        "current_wins": sum(game["winner"] == current_id for game in games),
        "opponent_wins": sum(game["winner"] == opponent_id for game in games),
        "tag_rate": len(valid_games) / len(games),
    }


def _plot_ratings(ratings: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(item["iteration"]) for item in ratings]
    figure, axis = plt.subplots(figsize=(12.5, 7.0))
    for key, label, color in (
        ("pursuer_elo", "Pursuer Elo", "#2563eb"),
        ("evader_elo", "Evader Elo", "#dc2626"),
        ("combined_elo", "Combined Elo", "#16a34a"),
    ):
        axis.plot(
            steps,
            [item[key] for item in ratings],
            marker="o",
            label=label,
            color=color,
        )
    axis.axhline(1000.0, color="#6b7280", linewidth=1.0, linestyle="--")
    axis.set_title(
        "Checkpoint Elo progression — valid TAG games only",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_xlabel("Checkpoint iteration")
    axis.set_ylabel("Role-conditioned Elo")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(config: Config) -> None:
    _validate_config(config)
    run_directory = config.run_directory.resolve()
    checkpoints = _checkpoint_paths(run_directory)
    run_manifest_path = run_directory / "manifest.json"
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path.is_file()
        else {}
    )
    training_config = run_manifest.get("config", {})
    reward_config = (
        tag_reward_config_from_dict(training_config["tag_reward"])
        if "tag_reward" in training_config
        else TagRewardConfig()
    )
    strategy_discount = training_config.get("strategy_discount", 1.0)
    _, p2_parameters = load_p2_checkpoint(config.locomotion_checkpoint)
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "historical", None)),
        config=MultiAgentTaskConfig(
            episode_duration=config.episode_duration,
            separation=config.initial_separation,
            minimum_separation=config.initial_separation,
            maximum_separation=config.initial_separation,
            naconmax=config.naconmax,
            njmax=config.njmax,
        ),
    )
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2_parameters[0]),
        HierarchicalRolloutConfig(
            stochastic_actions=config.action_mode == "stochastic",
            tag_reward=reward_config,
        ),
    )
    initial = runner.reset(jax.random.PRNGKey(config.seed))
    warm_start_parameters = migrate_p2_actor_parameters(
        runner.initialize_parameters(jax.random.PRNGKey(config.seed + 1), initial),
        p2_parameters[1],
    )
    parameters = {}
    manifests = {}
    for checkpoint in checkpoints:
        parameters[checkpoint.name], manifests[checkpoint.name] = _load_parameters(
            checkpoint, warm_start_parameters
        )

    ratings = TagCheckpointElo()
    pair_results = []
    rating_snapshots = []
    for current_index, current in enumerate(checkpoints):
        current_id = current.name
        for opponent in checkpoints[:current_index]:
            pair_results.append(
                _evaluate_pair(
                    current_id=current_id,
                    current_parameters=parameters[current_id],
                    opponent_id=opponent.name,
                    opponent_parameters=parameters[opponent.name],
                    runner=runner,
                    environment=environment,
                    config=config,
                    strategy_discount=strategy_discount,
                    ratings=ratings,
                )
            )
        snapshot = {
            "checkpoint": current_id,
            "iteration": int(manifests[current_id]["iteration"]),
            "state_sha256": manifests[current_id]["state_sha256"],
            "opponents_evaluated": current_index,
            **ratings.snapshot(current_id),
        }
        rating_snapshots.append(snapshot)

    output = config.output or (
        run_directory / f"checkpoint_elo_{config.action_mode}.json"
    )
    plot_output = output.with_suffix(".png")
    checks = {
        "all_games_have_finite_distances": all(
            np.isfinite(game["minimum_distance"])
            and np.isfinite(game["final_distance"])
            for pair in pair_results
            for game in pair["games"]
        ),
        "all_parameters_loaded": len(parameters) == len(checkpoints),
        "checkpoint_count": len(checkpoints) >= 2,
        "pair_count": len(pair_results)
        == len(checkpoints) * (len(checkpoints) - 1) // 2,
        "gpu_backend": jax.default_backend() == "gpu",
    }
    payload = {
        "backend": jax.default_backend(),
        "checkpoints": [
            {
                "checkpoint": checkpoint.name,
                "iteration": int(manifests[checkpoint.name]["iteration"]),
                "state_sha256": manifests[checkpoint.name]["state_sha256"],
            }
            for checkpoint in checkpoints
        ],
        "checks": checks,
        "config": {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(config).items()
            },
            "run_directory": str(run_directory),
            "output": str(output),
        },
        "experiment": "hierarchical_tag_checkpoint_matrix_elo",
        "evaluation_protocol": "role_conditioned_checkpoint_elo_v1",
        "action_mode": config.action_mode,
        "rating_method": {
            "initial_rating": ratings.initial_rating,
            "learning_rate": ratings.learning_rate,
            "role_bias_pursuer_win_probability": (
                ratings.baseline_pursuer_win_probability
            ),
            "valid_tag_games_only": True,
            "censored_outcomes_do_not_update_rating": True,
        },
        "pairs": pair_results,
        "ratings": rating_snapshots,
        "latest_checkpoint": rating_snapshots[-1],
        "plot": str(plot_output),
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "passed": all(checks.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload["latest_checkpoint_pairs"] = [
        pair
        for pair in pair_results
        if pair["current_checkpoint"] == checkpoints[-1].name
    ]
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_ratings(rating_snapshots, plot_output)
    print(json.dumps(payload, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
