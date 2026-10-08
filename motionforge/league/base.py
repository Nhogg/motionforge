"""Host-side opponent selection interfaces shared by league environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

OpponentCategory = Literal["recent", "historical", "best", "scripted"]


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
