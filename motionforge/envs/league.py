"""Environment layer that attaches league opponent selection to Warp physics."""

from __future__ import annotations

from motionforge.envs.warp import WarpEnv, WarpEnvConfig
from motionforge.league import League, LeagueOpponent


class LeagueEnv(WarpEnv):
    """Warp environment with one explicit host-side league dependency."""

    def __init__(
        self,
        *,
        league: League,
        config: WarpEnvConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        self.league: League = league

    def sample_opponent(self, *, seed: int) -> LeagueOpponent:
        """Select the frozen opponent for an upcoming rollout segment."""
        return self.league.sample_opponent(seed=seed)
