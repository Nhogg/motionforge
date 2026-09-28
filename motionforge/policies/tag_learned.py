"""Immutable learned TAG policies restored from reproducible PPO checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from brax.training import checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from motionforge.policies.tag_networks import (
    TAG_PURSUER_OBSERVATION_SIZE,
    make_tag_pursuer_ppo_networks,
)


def checkpoint_digest(path: Path) -> str:
    """Hash checkpoint paths and contents in a stable traversal order."""
    digest = hashlib.sha256()
    for entry in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(entry.relative_to(path).as_posix().encode())
        with entry.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenLearnedPursuer:
    """Versioned deterministic pursuer bound to one agent and checkpoint."""

    agent_index: int
    checkpoint_path: Path
    fixed_noise_std: float = 0.2
    specification_version: int = 1
    _checkpoint_digest: str = field(init=False, repr=False, compare=False)
    _parameters: Any = field(init=False, repr=False, compare=False)
    _policy: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.agent_index not in (0, 1):
            raise ValueError("agent_index must be 0 or 1")
        if self.fixed_noise_std <= 0.001:
            raise ValueError("fixed_noise_std must exceed 0.001")
        if self.specification_version <= 0:
            raise ValueError("specification_version must be positive")
        path = self.checkpoint_path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")

        networks = make_tag_pursuer_ppo_networks(
            running_statistics.normalize,
            fixed_noise_std=self.fixed_noise_std,
        )
        parameters = checkpoint.load(path)
        policy = ppo_networks.make_inference_fn(networks)(
            parameters,
            deterministic=True,
        )
        object.__setattr__(self, "checkpoint_path", path)
        object.__setattr__(self, "_checkpoint_digest", checkpoint_digest(path))
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(self, "_policy", policy)

    def command(self, policy_observation: jax.Array) -> jax.Array:
        """Return a deterministic physical velocity command."""
        if policy_observation.shape != (TAG_PURSUER_OBSERVATION_SIZE,):
            raise ValueError(
                "policy_observation must have shape "
                f"({TAG_PURSUER_OBSERVATION_SIZE},)"
            )
        normalized_action, _ = self._policy(
            policy_observation,
            jax.random.PRNGKey(0),
        )
        return jp.clip(normalized_action, -1.0, 1.0) * jp.asarray(
            [1.0, 0.5, 1.0]
        )

    def specification(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible opponent specification."""
        return {
            "agent_index": self.agent_index,
            "checkpoint_digest": self._checkpoint_digest,
            "fixed_noise_std": self.fixed_noise_std,
            "observation_size": TAG_PURSUER_OBSERVATION_SIZE,
            "policy_type": "learned_pursuer",
            "specification_version": self.specification_version,
        }

    @property
    def fingerprint(self) -> str:
        """Hash the canonical learned-opponent specification."""
        encoded = json.dumps(
            self.specification(), separators=(",", ":"), sort_keys=True
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FrozenLearnedEvader(FrozenLearnedPursuer):
    """Versioned deterministic evader with the shared TAG network contract."""

    def specification(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible opponent specification."""
        specification = super().specification()
        specification["policy_type"] = "learned_evader"
        return specification
