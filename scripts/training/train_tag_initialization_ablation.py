"""Isolated strategy command-head initialization experiment; unchanged PPO.

Only freshly initialized command kernels/biases are scaled. Encoders, value
heads, motor parameters, architecture, and restored checkpoints are unchanged.
Keep this experiment-specific adapter out of the production trainer.
"""

import math
from dataclasses import dataclass

import train_hierarchical_tag as base
from flax.core import freeze, unfreeze

from motionforge.cli import run_hydra


@dataclass
class Config(base.Config):
    initial_strategy_command_scale: float = 0.01


def scale_command_heads(parameters, scale):
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("initial_strategy_command_scale must be in (0,1]")
    parameters = unfreeze(parameters)
    for role in ("pursuer_head", "evader_head"):
        command = parameters["strategy"][role]["command"]
        for leaf in ("kernel", "bias"):
            command[leaf] = command[leaf] * scale
    return freeze(parameters)


def main(config):
    original = base.migrate_p2_actor_parameters

    def initialize(*args, **kwargs):
        return scale_command_heads(
            original(*args, **kwargs), config.initial_strategy_command_scale
        )

    base.migrate_p2_actor_parameters = initialize
    try:
        base.main(config)
    finally:
        base.migrate_p2_actor_parameters = original


if __name__ == "__main__":
    run_hydra(Config, main)
