"""brax_jax.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Compatibility adapters between the pinned Brax an current Jax versions.

    Brax 0.14.2 calls ``jax.device_put_replicated``, which JAX 0.10 removed.
    This module installs JAX's documented NameSharding-baesd replacement
    without modifying either dependency. Remove it once MotionForge pins a
    Brax release that supports the current JAX API directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def install_device_put_replicated_adapter() -> bool:
    """Install the removed JAX API when unavailable"""

    try:
        jax.device_put_replicated
    except AttributeError:
        pass
    else:
        return False

    def device_put_replicated(
        value: Any,
        devices: Sequence[jax.Device],
    ) -> Any:

        mesh = Mesh(
            np.asarray(devices),
            ("replica",),
        )
        sharding = NamedSharding(
            mesh,
            P("replica"),
        )

        return jax.tree.map(
            lambda leaf: jax.device_put(
                jnp.stack([leaf] * len(devices)),
                sharding,
            ),
            value,
        )

    jax.device_put_replicated = device_put_replicated
    return True
