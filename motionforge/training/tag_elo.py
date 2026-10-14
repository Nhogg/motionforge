"""Online role-specific Bradley--Terry/Elo ratings for TAG training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass
class TagEloTracker:
    """Track separate pursuer and evader ratings with structural role bias."""

    baseline_pursuer_win_probability: float = 0.65
    learning_rate: float = 24.0
    rating_scale: float = 400.0 / math.log(10.0)
    initial_rating: float = 1_000.0
    learner_pursuer_rating: float = 1_000.0
    learner_evader_rating: float = 1_000.0
    opponent_ratings: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.baseline_pursuer_win_probability < 1.0:
            raise ValueError("baseline pursuer win probability must be in (0, 1)")
        if self.learning_rate <= 0.0 or self.rating_scale <= 0.0:
            raise ValueError("Elo learning rate and rating scale must be positive")

    @property
    def role_bias(self) -> float:
        probability = self.baseline_pursuer_win_probability
        return math.log(probability / (1.0 - probability))

    def _opponent(self, opponent_id: str) -> dict[str, float]:
        return self.opponent_ratings.setdefault(
            opponent_id,
            {"pursuer": self.initial_rating, "evader": self.initial_rating},
        )

    def pursuer_win_probability(
        self,
        pursuer_rating: float,
        evader_rating: float,
    ) -> float:
        return _sigmoid(
            (pursuer_rating - evader_rating) / self.rating_scale + self.role_bias
        )

    def update(
        self,
        *,
        opponent_id: str,
        learner_pursuer_win_rate: float,
        opponent_pursuer_win_rate: float,
        pursuer_games: int,
        evader_games: int,
    ) -> dict[str, float]:
        """Update ratings from aggregate Bernoulli outcomes for both learner roles."""
        for value in (learner_pursuer_win_rate, opponent_pursuer_win_rate):
            if not 0.0 <= value <= 1.0:
                raise ValueError("TAG win rates must lie within [0, 1]")
        if pursuer_games <= 0 or evader_games <= 0:
            raise ValueError("both role matchups require at least one game")

        opponent = self._opponent(opponent_id)
        expected_pursuer = self.pursuer_win_probability(
            self.learner_pursuer_rating,
            opponent["evader"],
        )
        expected_opponent_pursuer = self.pursuer_win_probability(
            opponent["pursuer"],
            self.learner_evader_rating,
        )
        pursuer_delta = self.learning_rate * (
            learner_pursuer_win_rate - expected_pursuer
        )
        evader_delta = self.learning_rate * (
            opponent_pursuer_win_rate - expected_opponent_pursuer
        )
        self.learner_pursuer_rating += pursuer_delta
        opponent["evader"] -= pursuer_delta
        opponent["pursuer"] += evader_delta
        self.learner_evader_rating -= evader_delta

        return {
            **self.metrics(opponent_id),
            "elo/matchup/expected_learner_pursuer_win": expected_pursuer,
            "elo/matchup/expected_opponent_pursuer_win": expected_opponent_pursuer,
        }

    def metrics(self, opponent_id: str) -> dict[str, float]:
        """Report ratings without treating a censored trajectory as a game."""
        opponent = self._opponent(opponent_id)
        expected_pursuer = self.pursuer_win_probability(
            self.learner_pursuer_rating, opponent["evader"]
        )
        expected_opponent_pursuer = self.pursuer_win_probability(
            opponent["pursuer"], self.learner_evader_rating
        )

        pursuer_difficulty = 1.0 - self.pursuer_win_probability(
            self.learner_pursuer_rating,
            opponent["evader"],
        )
        evader_difficulty = self.pursuer_win_probability(
            opponent["pursuer"],
            self.learner_evader_rating,
        )
        pursuer_skill = _sigmoid(
            (self.learner_pursuer_rating - self.initial_rating) / self.rating_scale
        )
        evader_skill = _sigmoid(
            (self.learner_evader_rating - self.initial_rating) / self.rating_scale
        )
        return {
            "elo/role_bias_logit": self.role_bias,
            "elo/baseline_pursuer_win_probability": (
                self.baseline_pursuer_win_probability
            ),
            "elo/pursuer/rating": self.learner_pursuer_rating,
            "elo/pursuer/perceived_skill": pursuer_skill,
            "elo/pursuer/task_difficulty": pursuer_difficulty,
            "elo/evader/rating": self.learner_evader_rating,
            "elo/evader/perceived_skill": evader_skill,
            "elo/evader/task_difficulty": evader_difficulty,
            "elo/combined/rating": 0.5
            * (self.learner_pursuer_rating + self.learner_evader_rating),
            "elo/combined/perceived_skill": 0.5 * (pursuer_skill + evader_skill),
            "elo/combined/task_difficulty": 0.5
            * (pursuer_difficulty + evader_difficulty),
            "elo/matchup/expected_learner_pursuer_win": expected_pursuer,
            "elo/matchup/expected_opponent_pursuer_win": expected_opponent_pursuer,
        }
