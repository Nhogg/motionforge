"""Self-play population and scheduling infrastructure."""

from motionforge.self_play.population import (
    OpponentSelection,
    PolicySnapshot,
    select_opponent,
    save_policy_snapshot,
)

__all__ = [
    "OpponentSelection",
    "PolicySnapshot",
    "save_policy_snapshot",
    "select_opponent",
]
