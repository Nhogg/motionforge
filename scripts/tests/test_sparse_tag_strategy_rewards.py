"""Validate sparse hierarchical TAG payoffs and terminal GAE semantics."""

from __future__ import annotations

import json

import jax.numpy as jp
import numpy as np
from flax import struct

from motionforge.rollout import (
    AdvantageConfig,
    generalized_advantage_estimate,
    sparse_tag_strategy_reward,
)


@struct.dataclass
class TerminationFixture:
    tagged: object
    fallen: object
    out_of_bounds: object
    timed_out: object


def termination(
    *,
    tagged: bool = False,
    fallen: tuple[bool, bool] = (False, False),
    out_of_bounds: tuple[bool, bool] = (False, False),
    timed_out: bool = False,
) -> TerminationFixture:
    return TerminationFixture(
        tagged=jp.asarray(tagged),
        fallen=jp.asarray(fallen),
        out_of_bounds=jp.asarray(out_of_bounds),
        timed_out=jp.asarray(timed_out),
    )


def reward(fixture: TerminationFixture, *, pursuer: bool) -> float:
    learner_index = 0 if pursuer else 1
    return float(
        np.asarray(
            sparse_tag_strategy_reward(
                termination=fixture,
                learner_index=learner_index,
                learner_is_pursuer=pursuer,
            )
        )
    )


def main() -> None:
    tag = termination(tagged=True)
    evader_oob = termination(out_of_bounds=(False, True))
    pursuer_oob = termination(out_of_bounds=(True, False))
    timeout = termination(timed_out=True)
    fall = termination(fallen=(True, False))
    tag_and_oob = termination(tagged=True, out_of_bounds=(False, True))

    targets = generalized_advantage_estimate(
        rewards=jp.asarray([0.0, 0.0, 1.0]),
        values=jp.zeros(3),
        dones=jp.asarray([False, False, True]),
        valid=jp.ones(3, dtype=bool),
        bootstrap_value=jp.asarray(99.0),
        config=AdvantageConfig(discount=1.0, gae_lambda=1.0),
    )
    checks = {
        "pursuer_tag_is_one": np.isclose(reward(tag, pursuer=True), 1.0),
        "evader_tag_is_negative_one": np.isclose(reward(tag, pursuer=False), -1.0),
        "evader_exit_is_negative_one": np.isclose(
            reward(evader_oob, pursuer=False), -1.0
        ),
        "evader_exit_gives_pursuer_no_point": np.isclose(
            reward(evader_oob, pursuer=True), 0.0
        ),
        "pursuer_exit_is_neutral": np.isclose(reward(pursuer_oob, pursuer=True), 0.0)
        and np.isclose(reward(pursuer_oob, pursuer=False), 0.0),
        "timeout_is_neutral": np.isclose(reward(timeout, pursuer=True), 0.0)
        and np.isclose(reward(timeout, pursuer=False), 0.0),
        "fall_is_neutral": np.isclose(reward(fall, pursuer=True), 0.0),
        "duplicate_failure_is_single_loss": np.isclose(
            reward(tag_and_oob, pursuer=False), -1.0
        ),
        "terminal_bootstrap_is_zeroed": np.allclose(
            np.asarray(targets.returns), [1.0, 1.0, 1.0]
        ),
    }
    result = {
        "checks": {key: bool(value) for key, value in checks.items()},
        "experiment": "sparse_hierarchical_tag_strategy_rewards",
        "passed": bool(all(checks.values())),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
