"""Immutable TAG policy snapshots and their machine-readable population registry."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from motionforge.policies.tag_learned import checkpoint_digest

_VALID_ROLES = frozenset({"pursuer", "evader"})
_REGISTRY_VERSION = 1


@dataclass(frozen=True)
class PolicySnapshot:
    """One immutable checkpoint entry in a self-play population."""

    snapshot_id: str
    role: str
    generation: int
    checkpoint: str
    checkpoint_digest: str
    fixed_noise_std: float
    seed: int
    source_checkpoint: str


def _load_registry(path: Path) -> dict:
    if not path.exists():
        return {"registry_version": _REGISTRY_VERSION, "snapshots": []}
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("registry_version") != _REGISTRY_VERSION:
        raise ValueError(f"unsupported population registry: {path}")
    if not isinstance(registry.get("snapshots"), list):
        raise ValueError(f"invalid population registry: {path}")
    return registry


def _write_registry(path: Path, registry: dict) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_policy_snapshot(
    *,
    population_dir: Path,
    source_checkpoint: Path,
    role: str,
    generation: int,
    seed: int,
    fixed_noise_std: float,
) -> PolicySnapshot:
    """Copy one checkpoint into a population and atomically register it."""
    if role not in _VALID_ROLES:
        raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}")
    if generation < 0:
        raise ValueError("generation must be nonnegative")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if fixed_noise_std <= 0.001:
        raise ValueError("fixed_noise_std must exceed 0.001")

    source_checkpoint = source_checkpoint.resolve()
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {source_checkpoint}")

    population_dir = population_dir.resolve()
    population_dir.mkdir(parents=True, exist_ok=True)
    registry_path = population_dir / "population.json"
    registry = _load_registry(registry_path)
    if any(
        item["role"] == role and item["generation"] == generation
        for item in registry["snapshots"]
    ):
        raise FileExistsError(
            f"population already contains {role} generation {generation}"
        )

    digest = checkpoint_digest(source_checkpoint)
    snapshot_id = f"{role}_g{generation:04d}_{digest[:12]}"
    destination = population_dir / "checkpoints" / snapshot_id
    if destination.exists():
        raise FileExistsError(f"snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_name(f".{snapshot_id}.tmp")
    if temporary_destination.exists():
        raise FileExistsError(f"incomplete snapshot exists: {temporary_destination}")
    shutil.copytree(source_checkpoint, temporary_destination)
    if checkpoint_digest(temporary_destination) != digest:
        raise RuntimeError("copied checkpoint digest does not match source")
    temporary_destination.replace(destination)

    snapshot = PolicySnapshot(
        snapshot_id=snapshot_id,
        role=role,
        generation=generation,
        checkpoint=str(destination.relative_to(population_dir)),
        checkpoint_digest=digest,
        fixed_noise_std=fixed_noise_std,
        seed=seed,
        source_checkpoint=str(source_checkpoint),
    )
    registry["snapshots"].append(asdict(snapshot))
    registry["snapshots"].sort(key=lambda item: (item["generation"], item["role"]))
    _write_registry(registry_path, registry)
    return snapshot
