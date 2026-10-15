"""Isolated frozen-motor sampling ablation; strategy sampling remains unchanged.

With deterministic_frozen_locomotion=true, only the rollout motor sampler is
disabled. PPO architecture, losses, rewards, parameters and defaults are intact.
"""

from dataclasses import dataclass, replace

import train_hierarchical_tag as base

from motionforge.cli import run_hydra


@dataclass
class Config(base.Config):
    deterministic_frozen_locomotion: bool = False


def validate_motor_ablation(config: Config) -> None:
    if config.deterministic_frozen_locomotion and not config.freeze_locomotion:
        raise ValueError(
            "deterministic_frozen_locomotion requires freeze_locomotion=true"
        )


def main(config: Config) -> None:
    validate_motor_ablation(config)
    original = base.HierarchicalRolloutConfig

    def rollout_config(*args, **kwargs):
        configured = original(*args, **kwargs)
        return (
            replace(configured, stochastic_locomotion_actions=False)
            if config.deterministic_frozen_locomotion
            else configured
        )

    base.HierarchicalRolloutConfig = rollout_config
    try:
        base.main(config)
    finally:
        base.HierarchicalRolloutConfig = original


if __name__ == "__main__":
    run_hydra(Config, main)
