"""Validate deterministic weighted opponent sampling."""

from __future__ import annotations

import json
from collections import Counter

from motionforge.league import LeagueOpponent, WeightedLeague


def main() -> None:
    historical = tuple(
        LeagueOpponent(f"historical/{index}", "historical", {"id": index})
        for index in range(4)
    )
    auxiliary = (
        LeagueOpponent("stationary", "stationary", {"id": "stationary"}),
        LeagueOpponent(
            "scripted", "scripted", {"id": "scripted"}, controller="scripted"
        ),
        LeagueOpponent("current", "recent", {"id": "current"}),
    )
    opponents = historical + auxiliary
    expected = (0.1875,) * 4 + (1.0 / 12.0,) * 3
    league = WeightedLeague(opponents, expected)
    samples = 12_000
    selected = [league.sample_opponent(seed=seed) for seed in range(samples)]
    repeated = [league.sample_opponent(seed=seed) for seed in range(samples)]
    counts = Counter(opponent.opponent_id for opponent in selected)
    frequencies = {
        opponent.opponent_id: counts[opponent.opponent_id] / samples
        for opponent in opponents
    }
    category_counts = Counter(opponent.category for opponent in selected)
    category_frequencies = {
        category: count / samples for category, count in category_counts.items()
    }
    checks = {
        "deterministic": selected == repeated,
        "all_opponents_sampled": len(counts) == len(opponents),
        "frequencies_close": all(
            abs(frequencies[opponent.opponent_id] - probability) < 0.02
            for opponent, probability in zip(opponents, expected, strict=True)
        ),
        "historical_aggregate_close": abs(category_frequencies["historical"] - 0.75)
        < 0.02,
        "scripted_controller_preserved": auxiliary[1].controller == "scripted",
    }
    result = {
        "checks": checks,
        "experiment": "weighted_hierarchical_league",
        "category_frequencies": category_frequencies,
        "frequencies": frequencies,
        "passed": bool(all(checks.values())),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
