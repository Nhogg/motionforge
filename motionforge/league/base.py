"""Host-side opponent selection interfaces shared by league environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

OpponentCategory = Literal["recent", "historical", "best", "scripted", "warm_start"]


@dataclass(frozen=True)
class LeagueOpponent:
    """Exact immutable opponent selected for a rollout collection segment."""

    opponent_id: str
    category: OpponentCategory
    policy: Any


class League(ABC):
    """Host-side source of frozen or scripted rollout opponents."""

    @abstractmethod
    def sample_opponent(self, *, seed: int) -> LeagueOpponent:
        """Select one opponent deterministically from an explicit seed."""


class FixedLeague(League):
    """One-opponent league used for scripted play, evaluation, and tests."""

    def __init__(self, opponent: LeagueOpponent) -> None:
        self._opponent = opponent

    def sample_opponent(self, *, seed: int) -> LeagueOpponent:
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        return self._opponent


class HistoricalLeague(League):
    """Deterministically sample an immutable population of historical policies."""

    def __init__(self, opponents: Sequence[LeagueOpponent]) -> None:
        self._opponents = tuple(opponents)
        if not self._opponents:
            raise ValueError("historical league requires at least one opponent")
        if any(opponent.category != "historical" for opponent in self._opponents):
            raise ValueError("historical league opponents must be historical")
        opponent_ids = [opponent.opponent_id for opponent in self._opponents]
        if len(set(opponent_ids)) != len(opponent_ids):
            raise ValueError("historical league opponent IDs must be unique")

    @property
    def opponents(self) -> tuple[LeagueOpponent, ...]:
        return self._opponents

    def sample_opponent(self, *, seed: int) -> LeagueOpponent:
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        return self._opponents[seed % len(self._opponents)]
