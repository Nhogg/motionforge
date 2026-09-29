"""Self-play population and scheduling infrastructure."""

from motionforge.self_play.population import (
    OpponentSelection,
    PolicySnapshot,
    load_policy_snapshots,
    select_opponent,
    save_policy_snapshot,
)

__all__ = [
    "OpponentSelection",
    "PolicySnapshot",
    "load_policy_snapshots",
    "save_policy_snapshot",
    "select_opponent",
]
