from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tyro
from dm_control import viewer
from dm_control.locomotion import soccer


@dataclass(frozen=True)
class Config:
    """View the dm_control multi-agent soccer environment."""

    seed: int = 0
    team_size: int = 2
    walker_type: soccer.WalkerType = soccer.WalkerType.HUMANOID


def main(config: Config) -> None:
    environment_random_state = np.random.RandomState(config.seed)
    action_rng = np.random.default_rng(config.seed)

    environment = soccer.load(
        team_size=config.team_size,
        time_limit=45.0,
        random_state=environment_random_state,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=config.walker_type,
    )

    action_specs = environment.action_spec()

    def random_policy(time_step):
        del time_step

        return [
            action_rng.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
                size=action_spec.shape,
            )
            for action_spec in action_specs
        ]

    viewer.launch(
        environment_loader=lambda: environment,
        policy=random_policy,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
