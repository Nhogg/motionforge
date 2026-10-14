"""Regression checks for legacy and explicit recent-policy opponent shares."""

import math

from scripts.training.train_hierarchical_tag import mixed_opponent_probabilities


def main():
    legacy = mixed_opponent_probabilities(0.75)
    assert legacy["historical"] == 0.75
    assert legacy["current"] == legacy["stationary"] == legacy["scripted"] == 1 / 12
    recent = mixed_opponent_probabilities(0.25, 0.5)
    assert recent == {
        "historical": 0.25,
        "current": 0.5,
        "stationary": 0.125,
        "scripted": 0.125,
    }
    assert math.isclose(sum(legacy.values()), 1.0)
    assert sum(recent.values()) == 1.0
    for historical, current in [
        (0, None),
        (1, None),
        (0.75, 0.5),
        (0.25, 0),
        (0.25, float("nan")),
    ]:
        try:
            mixed_opponent_probabilities(historical, current)
        except ValueError:
            continue
        raise AssertionError((historical, current))
    print("Opponent mixture regression checks passed")


if __name__ == "__main__":
    main()
