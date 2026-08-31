from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tyro
from dm_control import viewer
from dm_control.locomotion.examples import basic_cmu_2019


@dataclass(frozen=True)
class Config:
    """View the official dm_control CMU humanoid example."""

    seed: int = 0


def main(config: Config) -> None:
    environment_random_state = np.random.RandomState(config.seed)
    action_rng = np.random.default_rng(config.seed)

    environment = basic_cmu_2019.cmu_humanoid_run_walls(
        random_state=environment_random_state
    )
    action_spec = environment.action_spec()

    def random_policy(time_step):
        del time_step

        return action_rng.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape,
        )

    viewer.launch(
        environment_loader=lambda: environment,
        policy=random_policy,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
