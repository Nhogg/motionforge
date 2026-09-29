"""Immutable TAG policy snapshots and their machine-readable population registry."""

from __future__ import annotations

import hashlib
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


@dataclass(frozen=True)
class OpponentSelection:
    """The exact current or historical opponent selected for one training job."""

    role: str
    source: str
    generation: int
    checkpoint: str
    checkpoint_digest: str
    fixed_noise_std: float
    snapshot_id: str | None
    seed: int
    current_probability: float


def _uniform_draw(*parts: object) -> float:
    encoded = ":".join(str(part) for part in parts).encode()
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return value / float(1 << 64)


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


def select_opponent(
    *,
    population_dir: Path,
    current_checkpoint: Path,
    role: str,
    current_generation: int,
    current_fixed_noise_std: float,
    seed: int,
    current_probability: float = 0.5,
) -> OpponentSelection:
    """Deterministically select a current or historical opponent for one job."""
    if role not in _VALID_ROLES:
        raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}")
    if current_generation < 0:
        raise ValueError("current_generation must be nonnegative")
    if current_fixed_noise_std <= 0.001:
        raise ValueError("current_fixed_noise_std must exceed 0.001")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not 0.0 <= current_probability <= 1.0:
        raise ValueError("current_probability must be in [0, 1]")

    population_dir = population_dir.resolve()
    current_checkpoint = current_checkpoint.resolve()
    if not current_checkpoint.is_dir():
        raise FileNotFoundError(
            f"current checkpoint does not exist: {current_checkpoint}"
        )
    current_digest = checkpoint_digest(current_checkpoint)
    registry = _load_registry(population_dir / "population.json")
    historical = [
        item
        for item in registry["snapshots"]
        if item["role"] == role and item["checkpoint_digest"] != current_digest
    ]
    historical.sort(key=lambda item: (item["generation"], item["snapshot_id"]))

    choose_current = not historical or _uniform_draw(
        "current", role, current_generation, seed
    ) < current_probability
    if choose_current:
        return OpponentSelection(
            role=role,
            source="current",
            generation=current_generation,
            checkpoint=str(current_checkpoint),
            checkpoint_digest=current_digest,
            fixed_noise_std=current_fixed_noise_std,
            snapshot_id=None,
            seed=seed,
            current_probability=current_probability,
        )

    index = min(
        int(
            _uniform_draw("historical", role, current_generation, seed)
            * len(historical)
        ),
        len(historical) - 1,
    )
    selected = historical[index]
    checkpoint = population_dir / selected["checkpoint"]
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"historical checkpoint does not exist: {checkpoint}")
    if checkpoint_digest(checkpoint) != selected["checkpoint_digest"]:
        raise RuntimeError(f"historical checkpoint digest mismatch: {checkpoint}")
    return OpponentSelection(
        role=role,
        source="historical",
        generation=selected["generation"],
        checkpoint=str(checkpoint),
        checkpoint_digest=selected["checkpoint_digest"],
        fixed_noise_std=selected["fixed_noise_std"],
        snapshot_id=selected["snapshot_id"],
        seed=seed,
        current_probability=current_probability,
    )
