"""Isolated strategy valid-sample weighting/normalization factorial experiment."""

from dataclasses import dataclass
from functools import partial

import train_hierarchical_tag as base

from motionforge.cli import run_hydra
from motionforge.training.tag_strategy_objective_ablation import (
    FullBatchStrategyObjectiveUpdater,
)


@dataclass
class Config(base.Config):
    strategy_accumulation_mode: str = "both"


def main(config):
    original = base.HierarchicalPpoUpdater
    base.HierarchicalPpoUpdater = partial(
        FullBatchStrategyObjectiveUpdater,
        accumulation_mode=config.strategy_accumulation_mode,
    )
    try:
        base.main(config)
    finally:
        base.HierarchicalPpoUpdater = original


if __name__ == "__main__":
    run_hydra(Config, main)
