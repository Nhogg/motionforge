"""Run the controlled hierarchical TAG diagnostic evaluation matrix."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from motionforge.cli import run_hydra


@dataclass(frozen=True)
class Evaluation:
    name: str
    checkpoint: Path
    opponent_checkpoint: Path | None
    stochastic_actions: bool


@dataclass
class Config:
    baseline_checkpoint: Path = Path(
        "logs/hierarchical/training/"
        "tag_joint_balanced_safe_full_episode_80k_seed0/checkpoints/000000000020"
    )
    historical_checkpoint: Path = Path(
        "logs/hierarchical/training/"
        "tag_joint_balanced_safe_80k_seed0/checkpoints/000000000010"
    )
    later_checkpoints: tuple[Path, ...] = (
        Path(
            "logs/hierarchical/training/"
            "tag_historical_1m_seed0_resume30/checkpoints/000000000145"
        ),
        Path(
            "logs/hierarchical/training/"
            "tag_historical_1m_seed1_resume30/checkpoints/000000000145"
        ),
        Path(
            "logs/hierarchical/training/"
            "tag_historical_1m_seed2_resume30/checkpoints/000000000145"
        ),
        Path(
            "logs/hierarchical/training/"
            "tag_historical_1m_seed3_resume30/checkpoints/000000000145"
        ),
    )
    seed: int = 20
    seeds: int = 16
    duration: float = 10.0
    gpus: tuple[int, ...] = tuple(range(8))
    resume: bool = True
    output_directory: Path = Path("logs/hierarchical/evaluation/diagnostics_overnight")
    plot_directory: Path = Path("logs/hierarchical/plots/diagnostics_overnight")


def _evaluations(config: Config) -> list[Evaluation]:
    evaluations = [
        Evaluation(
            "baseline_vs_warm_start_deterministic",
            config.baseline_checkpoint,
            None,
            False,
        ),
        Evaluation(
            "baseline_vs_warm_start_stochastic",
            config.baseline_checkpoint,
            None,
            True,
        ),
        Evaluation(
            "baseline_vs_historical_deterministic",
            config.baseline_checkpoint,
            config.historical_checkpoint,
            False,
        ),
        Evaluation(
            "baseline_vs_historical_stochastic",
            config.baseline_checkpoint,
            config.historical_checkpoint,
            True,
        ),
        Evaluation(
            "historical_vs_baseline_deterministic",
            config.historical_checkpoint,
            config.baseline_checkpoint,
            False,
        ),
        Evaluation(
            "historical_vs_baseline_stochastic",
            config.historical_checkpoint,
            config.baseline_checkpoint,
            True,
        ),
    ]
    for index, checkpoint in enumerate(config.later_checkpoints):
        evaluations.extend(
            (
                Evaluation(
                    f"later_seed{index}_vs_baseline_deterministic",
                    checkpoint,
                    config.baseline_checkpoint,
                    False,
                ),
                Evaluation(
                    f"baseline_vs_later_seed{index}_deterministic",
                    config.baseline_checkpoint,
                    checkpoint,
                    False,
                ),
            )
        )
    evaluations.extend(
        (
            Evaluation(
                "later_seed0_vs_baseline_stochastic",
                config.later_checkpoints[0],
                config.baseline_checkpoint,
                True,
            ),
            Evaluation(
                "baseline_vs_later_seed0_stochastic",
                config.baseline_checkpoint,
                config.later_checkpoints[0],
                True,
            ),
        )
    )
    return evaluations


def _validate(config: Config, evaluations: list[Evaluation]) -> None:
    if not config.gpus:
        raise ValueError("at least one GPU is required")
    if config.seeds <= 0 or config.duration <= 0.0:
        raise ValueError("seeds and duration must be positive")
    checkpoints = {
        path
        for evaluation in evaluations
        for path in (evaluation.checkpoint, evaluation.opponent_checkpoint)
        if path is not None
    }
    missing = [str(path) for path in checkpoints if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")


def _command(config: Config, evaluation: Evaluation) -> list[str]:
    command = [
        sys.executable,
        "scripts/evaluation/evaluate_hierarchical_tag.py",
        f"checkpoint={evaluation.checkpoint}",
        f"seed={config.seed}",
        f"seeds={config.seeds}",
        f"duration={config.duration}",
        f"stochastic_actions={str(evaluation.stochastic_actions).lower()}",
        "record_trajectories=true",
        f"output={config.output_directory / (evaluation.name + '.json')}",
    ]
    if evaluation.opponent_checkpoint is not None:
        command.append(f"opponent_checkpoint={evaluation.opponent_checkpoint}")
    return command


def main(config: Config) -> None:
    evaluations = _evaluations(config)
    _validate(config, evaluations)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[Evaluation, int, subprocess.Popen[str], object]] = []
    for index, evaluation in enumerate(evaluations):
        output_path = config.output_directory / f"{evaluation.name}.json"
        if config.resume and output_path.is_file():
            print(
                json.dumps({"event": "skipped", "name": evaluation.name}),
                flush=True,
            )
            continue
        gpu = config.gpus[index % len(config.gpus)]
        log_path = config.output_directory / f"{evaluation.name}.log"
        stream = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        process = subprocess.Popen(
            _command(config, evaluation),
            cwd=Path.cwd(),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((evaluation, gpu, process, stream))
        print(
            json.dumps({"event": "started", "gpu": gpu, "name": evaluation.name}),
            flush=True,
        )

    failures = []
    try:
        for evaluation, gpu, process, stream in processes:
            return_code = process.wait()
            stream.close()
            event = {
                "event": "completed" if return_code == 0 else "failed",
                "gpu": gpu,
                "name": evaluation.name,
                "return_code": return_code,
            }
            print(json.dumps(event), flush=True)
            if return_code != 0:
                failures.append(event)
    except BaseException:
        for _, _, process, stream in processes:
            if process.poll() is None:
                process.terminate()
            stream.close()
        raise
    if failures:
        raise RuntimeError(f"diagnostic evaluations failed: {failures}")

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/plot_hierarchical_tag.py",
            f"evaluation_directory={config.output_directory}",
            "pattern=*.json",
            "trajectory_pattern=*.json",
            f"output_directory={config.plot_directory}",
        ],
        check=True,
        cwd=Path.cwd(),
    )
    print(
        json.dumps(
            {
                "evaluation_count": len(evaluations),
                "event": "diagnostics_complete",
                "output_directory": str(config.output_directory),
                "plot_directory": str(config.plot_directory),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run_hydra(Config, main)
