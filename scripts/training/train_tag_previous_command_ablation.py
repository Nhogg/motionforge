"""Fresh learner-only previous-command parameter-constraint experiment.

Control false uses the original updater/initializer exactly. Motor PPO remains
frozen in both study arms. No global observation masking or architecture change.
"""

from dataclasses import dataclass

import train_hierarchical_tag as base

from motionforge.cli import run_hydra
from motionforge.training.tag_previous_command_ablation import (
    PreviousCommandConstrainedUpdater,
    constrain_hierarchical_previous_command,
)


@dataclass
class Config(base.Config):
    mask_previous_command: bool = True


def validate_constraint(config):
    if not config.freeze_locomotion:
        raise ValueError("The previous-command study requires freeze_locomotion=true")
    if config.mask_previous_command and config.restore_checkpoint is not None:
        raise ValueError(
            "mask_previous_command requires fresh initialization, not restore_checkpoint"
        )


def main(config):
    validate_constraint(config)
    if not config.mask_previous_command:
        return base.main(config)
    original_initializer = base.migrate_p2_actor_parameters
    original_updater = base.HierarchicalPpoUpdater

    def initialize(*args, **kwargs):
        return constrain_hierarchical_previous_command(
            original_initializer(*args, **kwargs)
        )

    base.migrate_p2_actor_parameters = initialize
    base.HierarchicalPpoUpdater = PreviousCommandConstrainedUpdater
    try:
        return base.main(config)
    finally:
        base.migrate_p2_actor_parameters = original_initializer
        base.HierarchicalPpoUpdater = original_updater


if __name__ == "__main__":
    run_hydra(Config, main)
