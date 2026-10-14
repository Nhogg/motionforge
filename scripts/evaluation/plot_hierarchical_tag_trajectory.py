"""Plot world-space TAG paths, separation, and each agent's radial movement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.envs.tag_endings import tag_ending_metadata
from motionforge.evaluation.hierarchical_roles import (
    audit_hierarchical_roles,
)
from motionforge.evaluation.hierarchical_trajectory import (
    plot_path,
    trajectory_diagnostics,
)
from motionforge.league import FixedLeague, LeagueOpponent
from motionforge.models import (
    load_p2_checkpoint,
    migrate_p2_actor_parameters,
    p2_locomotion_normalization,
)
from motionforge.rollout import HierarchicalRolloutConfig, HierarchicalRolloutRunner
from motionforge.training import (
    HierarchicalPpoUpdater,
    HierarchicalUpdateConfig,
    load_hierarchical_checkpoint,
    read_hierarchical_checkpoint_manifest,
)


@dataclass
class Config:
    checkpoint: Path = Path("???")
    opponent_checkpoint: Path | None = None
    use_self_play: bool = True
    role: str = "pursuer"
    seed: int = 0
    stochastic_actions: bool = False
    use_training_reset: bool = True
    initial_separation: float | None = None
    duration: float = 10.0
    naconmax: int = 256
    njmax: int = 4096
    boundary_command_safety_enabled: bool = True
    boundary_command_margin: float = 1.0
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    output: Path = Path("logs/hierarchical/trajectories/checkpoint_path.png")


def _load(checkpoint, warm_start):
    manifest = read_hierarchical_checkpoint_manifest(checkpoint)
    updater = HierarchicalPpoUpdater(
        HierarchicalUpdateConfig(**manifest["update_config"])
    )
    return load_hierarchical_checkpoint(checkpoint, updater.initialize(warm_start))


def main(config: Config) -> None:
    if config.role not in {"pursuer", "evader"} or config.duration <= 0:
        raise ValueError("require role=pursuer/evader and positive duration")
    if config.initial_separation is not None and config.initial_separation <= 0:
        raise ValueError("initial_separation must be positive")
    training_config = {}
    training_manifest = config.checkpoint.parent.parent / "manifest.json"
    if config.use_training_reset and training_manifest.is_file():
        training_config = json.loads(training_manifest.read_text()).get("config", {})
    minimum = training_config.get("minimum_initial_separation", 2.0)
    maximum = training_config.get("maximum_initial_separation", minimum)
    if config.initial_separation is not None:
        minimum = maximum = config.initial_separation
    _, p2 = load_p2_checkpoint(config.locomotion_checkpoint)
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None)),
        config=MultiAgentTaskConfig(
            separation=minimum,
            minimum_separation=minimum,
            maximum_separation=maximum,
            episode_duration=config.duration,
            naconmax=config.naconmax,
            njmax=config.njmax,
        ),
    )
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2[0]),
        HierarchicalRolloutConfig(
            stochastic_actions=config.stochastic_actions,
            boundary_command_safety_enabled=config.boundary_command_safety_enabled,
            boundary_command_margin=config.boundary_command_margin,
        ),
    )
    state = runner.reset(jax.random.PRNGKey(config.seed))
    warm_start = migrate_p2_actor_parameters(
        runner.initialize_parameters(jax.random.PRNGKey(config.seed + 1), state), p2[1]
    )
    learner, manifest = _load(config.checkpoint, warm_start)
    if config.opponent_checkpoint is not None:
        opponent, _ = _load(config.opponent_checkpoint, warm_start)
        opponent_parameters = opponent.parameters
        opponent_id, category = str(config.opponent_checkpoint.resolve()), "historical"
    elif config.use_self_play:
        opponent_parameters = learner.parameters
        opponent_id, category = "current/self-play", "recent"
    else:
        opponent_parameters = warm_start
        opponent_id, category = "warm-start/p2", "warm_start"
    environment.league = FixedLeague(
        LeagueOpponent(opponent_id, category, opponent_parameters)
    )
    segment = runner.start_segment(
        learner.parameters,
        seed=config.seed,
        learner_is_pursuer=config.role == "pursuer",
    )
    audit = audit_hierarchical_roles(
        pursuer_index=environment.config.pursuer_index,
        segment=segment,
        roles=state.roles,
    )
    positions = [np.asarray(state.environment.diagnostics.planar_position)]
    commands = [np.asarray(state.commands)]
    roles = [np.asarray(state.roles)]
    step = jax.jit(runner.step)
    dt = environment.config.control_timestep
    for _ in range(round(config.duration / dt)):
        state = step(segment, state)
        positions.append(np.asarray(state.environment.diagnostics.planar_position))
        commands.append(np.asarray(state.commands))
        roles.append(np.asarray(state.roles))
        if bool(np.asarray(state.environment.done)):
            break
    if not np.array_equal(
        np.asarray(roles), np.broadcast_to(roles[0], np.shape(roles))
    ):
        raise RuntimeError("physical-agent role encodings changed during rollout")
    diagnostics = trajectory_diagnostics(positions, dt, audit)
    diagnostics["trajectory"]["executed_commands_heading_frame"] = np.asarray(
        commands
    ).tolist()
    termination = state.environment.termination
    payload = {
        **audit,
        **diagnostics,
        "checkpoint": str(config.checkpoint.resolve()),
        "checkpoint_iteration": int(learner.iteration),
        "legacy_strategy_head_migrated": bool(
            manifest.get("legacy_strategy_head_migrated", False)
        ),
        "checkpoint_environment_steps": manifest.get("metadata", {}).get(
            "environment_steps"
        ),
        "opponent_id": opponent_id,
        "seed": config.seed,
        "stochastic_actions": config.stochastic_actions,
        "initial_separation_range": [minimum, maximum],
        "arena_half_extent": environment.config.arena_half_extent,
        "termination": {
            **tag_ending_metadata(termination, collection_cutoff=True),
            "tagged": bool(np.asarray(termination.tagged)),
            "fallen": np.asarray(termination.fallen).tolist(),
            "out_of_bounds": np.asarray(termination.out_of_bounds).tolist(),
            "timed_out": bool(np.asarray(termination.timed_out)),
        },
    }
    plot_path(payload, config.output)
    config.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "trajectory"}))


if __name__ == "__main__":
    run_hydra(Config, main)
