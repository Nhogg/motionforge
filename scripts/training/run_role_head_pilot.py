"""Queue, train, and evaluate the four-seed role-specific-head pilot."""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
PYTHON = REPOSITORY / ".venv/bin/python"
RESTORE = Path(
    "logs/hierarchical/training/"
    "tag_joint_balanced_safe_full_episode_80k_seed0/checkpoints/000000000020"
)
HISTORICAL = (
    Path(
        "logs/hierarchical/training/"
        "tag_joint_balanced_safe_80k_seed0/checkpoints/000000000010"
    ),
    RESTORE,
)
TRAINING_ROOT = Path("logs/hierarchical/training/role_heads_pilot")
EVALUATION_ROOT = Path("logs/hierarchical/evaluation/role_heads_pilot")
PLOT_ROOT = Path("logs/hierarchical/plots/role_heads_pilot")
SEEDS = (0, 1, 2, 3)


def _available_gpus(required: int = 4, maximum_used_mib: int = 4096) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    available = []
    for line in result.stdout.splitlines():
        index, used = (int(value.strip()) for value in line.split(","))
        if used <= maximum_used_mib:
            available.append(index)
    requested = os.environ.get("MOTIONFORGE_PILOT_GPUS")
    if requested:
        preferred = [int(value) for value in requested.split(",")]
        return [index for index in preferred if index in available][:required]
    return available[:required]


def _run(command: list[str], gpu: int, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=REPOSITORY,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _train(seed: int, gpu: int) -> Path:
    output = TRAINING_ROOT / f"seed{seed}"
    checkpoints_directory = output / "checkpoints"
    if (output / "summary.json").is_file() and checkpoints_directory.is_dir():
        checkpoints = sorted(checkpoints_directory.iterdir())
        if checkpoints:
            return checkpoints[-1]
    historical = ",".join(str(path) for path in HISTORICAL)
    command = [
        str(PYTHON),
        "scripts/training/train_hierarchical_tag.py",
        f"restore_checkpoint={RESTORE}",
        f"historical_opponent_checkpoints=[{historical}]",
        f"seed={seed}",
        "num_environments=16",
        "num_iterations=20",
        "rollout_horizon=500",
        "episode_duration=10.0",
        "checkpoint_interval=5",
        "wandb_mode=online",
        "wandb_group=hierarchical_tag_role_heads_pilot",
        f"wandb_name=role_heads_pilot_seed{seed}",
        f"output_dir={output}",
    ]
    _run(command, gpu, output.with_suffix(".log"))
    checkpoints = sorted((output / "checkpoints").iterdir())
    if not checkpoints:
        raise RuntimeError(f"seed {seed} produced no checkpoints")
    return checkpoints[-1]


def _evaluate(seed: int, gpu: int, checkpoint: Path) -> None:
    for opponent_name, opponent in (
        ("iter10", HISTORICAL[0]),
        ("iter20", HISTORICAL[1]),
    ):
        for mode, stochastic in (("deterministic", False), ("stochastic", True)):
            stem = f"seed{seed}_vs_{opponent_name}_{mode}"
            output = EVALUATION_ROOT / f"{stem}.json"
            if output.is_file():
                continue
            command = [
                str(PYTHON),
                "scripts/evaluation/evaluate_hierarchical_tag.py",
                f"checkpoint={checkpoint}",
                f"opponent_checkpoint={opponent}",
                "seed=20",
                "seeds=16",
                "duration=10.0",
                f"stochastic_actions={str(stochastic).lower()}",
                "record_trajectories=true",
                f"output={output}",
            ]
            _run(command, gpu, EVALUATION_ROOT / f"{stem}.log")


def main() -> None:
    if not RESTORE.is_dir() or any(not path.is_dir() for path in HISTORICAL):
        raise FileNotFoundError("one or more pilot checkpoints are missing")
    while len(gpus := _available_gpus()) < len(SEEDS):
        print(
            json.dumps(
                {
                    "event": "waiting_for_gpus",
                    "free_gpus": gpus,
                    "required": len(SEEDS),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(30)

    assignments = dict(zip(SEEDS, gpus, strict=True))
    print(json.dumps({"event": "training_started", "assignments": assignments}))
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        checkpoints = dict(
            zip(
                SEEDS,
                executor.map(lambda seed: _train(seed, assignments[seed]), SEEDS),
                strict=True,
            )
        )
    print(
        json.dumps(
            {
                "event": "training_completed",
                "checkpoints": {seed: str(path) for seed, path in checkpoints.items()},
            }
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        list(
            executor.map(
                lambda seed: _evaluate(seed, assignments[seed], checkpoints[seed]),
                SEEDS,
            )
        )
    subprocess.run(
        [
            str(PYTHON),
            "scripts/evaluation/plot_hierarchical_tag.py",
            f"evaluation_directory={EVALUATION_ROOT}",
            "pattern=*.json",
            "trajectory_pattern=*.json",
            f"output_directory={PLOT_ROOT}",
        ],
        cwd=REPOSITORY,
        check=True,
    )
    print(json.dumps({"event": "pilot_completed", "plots": str(PLOT_ROOT)}))


if __name__ == "__main__":
    main()
