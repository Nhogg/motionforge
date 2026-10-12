"""Audit deterministic immutable sampling from a historical policy league."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from motionforge.cli import run_hydra
from motionforge.league import HistoricalLeague, LeagueOpponent


@dataclass
class Config:
    samples: int = 9
    output: Path = Path("logs/hierarchical/historical_league_a.json")


def main(config: Config) -> None:
    if config.samples < 3:
        raise ValueError("samples must be at least three")
    policies = tuple({"generation": generation} for generation in range(3))
    opponents = tuple(
        LeagueOpponent(
            opponent_id=f"historical/generation-{generation:04d}",
            category="historical",
            policy=policy,
        )
        for generation, policy in enumerate(policies)
    )
    league = HistoricalLeague(opponents)
    sampled = [league.sample_opponent(seed=seed) for seed in range(config.samples)]
    repeated = [league.sample_opponent(seed=seed) for seed in range(config.samples)]

    invalid_category_rejected = False
    try:
        HistoricalLeague([LeagueOpponent("warm-start", "warm_start", policies[0])])
    except ValueError:
        invalid_category_rejected = True
    duplicate_id_rejected = False
    try:
        HistoricalLeague([opponents[0], opponents[0]])
    except ValueError:
        duplicate_id_rejected = True

    checks = {
        "all_opponents_sampled": {opponent.opponent_id for opponent in sampled}
        == {opponent.opponent_id for opponent in opponents},
        "deterministic": sampled == repeated,
        "duplicate_id_rejected": duplicate_id_rejected,
        "invalid_category_rejected": invalid_category_rejected,
        "policies_preserved_by_identity": all(
            opponent.policy is policies[index % len(policies)]
            for index, opponent in enumerate(sampled)
        ),
        "seed_cycles_population": all(
            opponent is opponents[index % len(opponents)]
            for index, opponent in enumerate(sampled)
        ),
    }
    result = {
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "hierarchical_historical_league",
        "opponent_ids": [opponent.opponent_id for opponent in sampled],
        "passed": all(checks.values()),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
