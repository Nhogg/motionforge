"""Policy transformations for hierarchical curriculum opponents."""

from __future__ import annotations

from typing import Any

import jax.numpy as jp
from flax.core import freeze, unfreeze


def make_stationary_hierarchical_opponent(
    parameters: Any,
    *,
    strategy_log_std: float = -20.0,
) -> Any:
    """Return a policy that holds a near-deterministic zero strategy command."""
    mutable = unfreeze(parameters)
    try:
        strategy = mutable["strategy"]
        role_heads = (strategy["pursuer_head"], strategy["evader_head"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "parameters do not contain both strategy role heads"
        ) from error

    for head in role_heads:
        command = head["command"]
        command["kernel"] = jp.zeros_like(command["kernel"])
        command["bias"] = jp.zeros_like(command["bias"])
        head["log_std"] = jp.full_like(head["log_std"], strategy_log_std)
    return freeze(mutable)
