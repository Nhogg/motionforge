"""Select the strongest role-conservative hierarchical TAG checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.evaluation import select_hierarchical_checkpoint


@dataclass
class Config:
    evaluation_directory: Path = Path("logs/hierarchical/evaluation")
    pattern: str = "tag_joint_*.json"
    output: Path = Path("logs/hierarchical/evaluation/checkpoint_selection.json")


def main(config: Config) -> None:
    if not config.evaluation_directory.is_dir():
        raise FileNotFoundError(
            f"evaluation directory does not exist: {config.evaluation_directory}"
        )
    paths = sorted(config.evaluation_directory.glob(config.pattern))
    if not paths:
        raise FileNotFoundError(
            f"no evaluation reports match {config.pattern!r} in "
            f"{config.evaluation_directory}"
        )
    evaluations = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    selected, ranked = select_hierarchical_checkpoint(evaluations)
    result = {
        "candidate_count": len(ranked),
        "config": {
            **asdict(config),
            "evaluation_directory": str(config.evaluation_directory),
            "output": str(config.output),
        },
        "experiment": "hierarchical_tag_checkpoint_selection",
        "ranking_order": [candidate.as_dict() for candidate in ranked],
        "selected": selected.as_dict(),
        "selection_rule": [
            "maximize minimum role success",
            "maximize minimum physical survival",
            "maximize mean role success",
            "maximize mean physical survival",
            "minimize maximum command-tracking RMSE",
            "prefer earlier iteration on an exact tie",
        ],
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run_hydra(Config, main)
