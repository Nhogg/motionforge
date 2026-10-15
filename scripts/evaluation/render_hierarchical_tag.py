"""Render a deterministic hierarchical TAG checkpoint rollout."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio_ffmpeg
import jax
import mediapy
import mujoco
import numpy as np
from mujoco import mjx

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.envs.tag_endings import tag_ending_metadata
from motionforge.evaluation.hierarchical_roles import (
    annotate_hierarchical_frame,
    audit_hierarchical_roles,
    audit_hierarchical_role_trace,
    color_hierarchical_agents,
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
    use_self_play: bool = False
    stochastic_actions: bool = False
    initial_separation: float | None = None
    use_training_reset: bool = False
    locomotion_checkpoint: Path = Path(
        "logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320"
    )
    role: str = "pursuer"
    seed: int = 0
    duration: float = 10.0
    fps: int = 30
    width: int = 960
    height: int = 720
    camera_distance: float = 5.0
    naconmax: int = 256
    njmax: int = 512
    boundary_command_safety_enabled: bool = True
    boundary_command_margin: float = 1.0
    output: Path = Path("logs/hierarchical/videos/tag_checkpoint_pursuer_a.mp4")


def _validate_config(config: Config) -> None:
    if config.initial_separation is not None and config.initial_separation <= 0:
        raise ValueError("initial_separation must be positive")
    if config.use_self_play and config.opponent_checkpoint is not None:
        raise ValueError("choose use_self_play or opponent_checkpoint")
    if (
        config.opponent_checkpoint is not None
        and not config.opponent_checkpoint.is_dir()
    ):
        raise FileNotFoundError(config.opponent_checkpoint)
    if config.role not in {"pursuer", "evader"}:
        raise ValueError("role must be pursuer or evader")
    if config.duration <= 0.0:
        raise ValueError("duration must be positive")
    if config.fps <= 0 or config.width <= 0 or config.height <= 0:
        raise ValueError("fps, width, and height must be positive")
    if config.camera_distance <= 0.0:
        raise ValueError("camera_distance must be positive")
    if config.naconmax <= 0 or config.njmax <= 0:
        raise ValueError("naconmax and njmax must be positive")
    if config.boundary_command_margin <= 0.0:
        raise ValueError("boundary_command_margin must be positive")
    if not config.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {config.checkpoint}")
    if not config.locomotion_checkpoint.is_dir():
        raise FileNotFoundError(
            f"locomotion checkpoint does not exist: {config.locomotion_checkpoint}"
        )


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Encode with the self-contained ffmpeg distributed by imageio-ffmpeg."""
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(path, frames, fps=fps)


def main(config: Config) -> None:
    _validate_config(config)
    manifest = read_hierarchical_checkpoint_manifest(config.checkpoint)
    update_config = HierarchicalUpdateConfig(**manifest["update_config"])
    _, p2_parameters = load_p2_checkpoint(config.locomotion_checkpoint)
    minimum_separation = maximum_separation = 2.0
    training_manifest = config.checkpoint.parent.parent / "manifest.json"
    if config.use_training_reset and training_manifest.is_file():
        training_config = json.loads(training_manifest.read_text()).get("config", {})
        minimum_separation = training_config.get("minimum_initial_separation", 2.0)
        maximum_separation = training_config.get(
            "maximum_initial_separation", minimum_separation
        )
    if config.initial_separation is not None:
        minimum_separation = maximum_separation = config.initial_separation
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("placeholder", "scripted", None)),
        config=MultiAgentTaskConfig(
            separation=minimum_separation,
            minimum_separation=minimum_separation,
            maximum_separation=maximum_separation,
            episode_duration=config.duration,
            naconmax=config.naconmax,
            njmax=config.njmax,
        ),
    )
    runner = HierarchicalRolloutRunner(
        environment,
        p2_locomotion_normalization(p2_parameters[0]),
        HierarchicalRolloutConfig(
            stochastic_actions=config.stochastic_actions,
            boundary_command_safety_enabled=(config.boundary_command_safety_enabled),
            boundary_command_margin=config.boundary_command_margin,
        ),
    )
    state = runner.reset(jax.random.PRNGKey(config.seed))
    warm_start_parameters = migrate_p2_actor_parameters(
        runner.initialize_parameters(jax.random.PRNGKey(config.seed + 1), state),
        p2_parameters[1],
    )
    template = HierarchicalPpoUpdater(update_config).initialize(warm_start_parameters)
    train_state, _ = load_hierarchical_checkpoint(config.checkpoint, template)
    opponent_parameters = warm_start_parameters
    opponent_id, opponent_category = "warm-start/p2", "warm_start"
    if config.use_self_play:
        opponent_parameters = train_state.parameters
        opponent_id, opponent_category = "current/self-play", "recent"
    elif config.opponent_checkpoint is not None:
        opponent_state, _ = load_hierarchical_checkpoint(
            config.opponent_checkpoint, template
        )
        opponent_parameters = opponent_state.parameters
        opponent_id, opponent_category = (
            str(config.opponent_checkpoint.resolve()),
            "historical",
        )
    environment.league = FixedLeague(
        LeagueOpponent(opponent_id, opponent_category, opponent_parameters)
    )
    segment = runner.start_segment(
        train_state.parameters,
        seed=config.seed,
        learner_is_pursuer=config.role == "pursuer",
    )
    audit = audit_hierarchical_roles(
        pursuer_index=environment.config.pursuer_index,
        segment=segment,
        roles=state.roles,
    )
    step = jax.jit(runner.step)

    model = environment.model_bundle.model
    colored_geom_counts = color_hierarchical_agents(
        environment.model_bundle, pursuer_index=audit["pursuer_index"]
    )
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)

    renderer = mujoco.Renderer(model, width=config.width, height=config.height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = config.camera_distance
    camera.azimuth = 135.0
    camera.elevation = -20.0
    frames: list[np.ndarray] = []
    next_frame_time = 0.0
    terminated_at = None
    minimum_distance = float("inf")
    minimum_root_height = float("inf")
    maximum_command_norm = 0.0
    rollout_steps = round(config.duration / environment.config.control_timestep)
    role_history = [np.asarray(state.roles)]
    try:
        for step_index in range(rollout_steps):
            state = step(segment, state)
            state.environment.physics.data.qpos.block_until_ready()
            role_history.append(np.asarray(state.roles))
            frame_time = (step_index + 1) * environment.config.control_timestep
            positions = np.asarray(state.environment.diagnostics.planar_position)
            distance = float(np.linalg.norm(positions[1] - positions[0]))
            minimum_distance = min(minimum_distance, distance)
            minimum_root_height = min(
                minimum_root_height,
                float(np.asarray(state.environment.diagnostics.root_height).min()),
            )
            maximum_command_norm = max(
                maximum_command_norm,
                float(np.linalg.norm(np.asarray(state.commands), axis=-1).max()),
            )
            if frame_time >= next_frame_time:
                camera.lookat[:2] = positions.mean(axis=0)
                camera.lookat[2] = 0.7
                host_data = mjx.get_data(model, state.environment.physics.data)
                renderer.update_scene(host_data, camera=camera)
                frames.append(
                    annotate_hierarchical_frame(
                        renderer.render(),
                        audit,
                        stochastic_actions=config.stochastic_actions,
                        matchup=opponent_id,
                    )
                )
                next_frame_time += 1.0 / config.fps
            if bool(np.asarray(state.environment.done)):
                terminated_at = frame_time
                break
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("no video frames were captured")
    role_trace_audit = audit_hierarchical_role_trace(
        np.stack(role_history),
        pursuer_index=audit["pursuer_index"],
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    _write_video(config.output, frames, config.fps)
    termination = state.environment.termination
    payload = {
        **audit,
        **role_trace_audit,
        "colored_geom_counts": colored_geom_counts,
        "opponent_id": opponent_id,
        "initial_separation_range": [minimum_separation, maximum_separation],
        "backend": jax.default_backend(),
        "checkpoint": str(config.checkpoint.resolve()),
        "checkpoint_iteration": int(train_state.iteration),
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "opponent_checkpoint": str(config.opponent_checkpoint)
            if config.opponent_checkpoint
            else None,
            "locomotion_checkpoint": str(config.locomotion_checkpoint),
            "output": str(config.output),
        },
        "experiment": "hierarchical_tag_checkpoint_video",
        "frames": len(frames),
        "learner_index": segment.learner_index,
        "maximum_command_norm": maximum_command_norm,
        "minimum_agent_distance": minimum_distance,
        "minimum_root_height": minimum_root_height,
        "mujoco_version": mujoco.__version__,
        "output": str(config.output.resolve()),
        "python_version": platform.python_version(),
        "role": config.role,
        "terminated_at": terminated_at,
        "termination": {
            **tag_ending_metadata(termination, collection_cutoff=True),
            "fallen": np.asarray(termination.fallen).tolist(),
            "out_of_bounds": np.asarray(termination.out_of_bounds).tolist(),
            "tagged": bool(np.asarray(termination.tagged)),
            "timed_out": bool(np.asarray(termination.timed_out)),
        },
        "video_duration": len(frames) / config.fps,
    }
    config.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
