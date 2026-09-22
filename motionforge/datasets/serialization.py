"""Host-side serialization helpers shared by locomotion dataset generators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Self

import jax
import numpy as np

from motionforge.datasets.locomotion import LOCOMOTION_DATASET_SCHEMA_VERSION


class JsonlDatasetWriter:
    """Stream records to an atomic JSONL artifact with an incremental digest."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.temporary_output = output.with_suffix(f"{output.suffix}.tmp")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.temporary_output.open("w", encoding="utf-8")
        self._digest = hashlib.sha256()
        self.count = 0

    def write(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        self._stream.write(line)
        self._digest.update(line.encode())
        self.count += 1

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def close(self) -> None:
        if self._stream.closed:
            return
        self._stream.close()
        self.temporary_output.replace(self.output)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._stream.close()


def locomotion_record(
    sample,
    *,
    dataset_kind: str,
    episode: int,
    episode_seed: int,
    timestep: int,
    state,
    extra: dict | None = None,
) -> dict:
    """Convert one device sample to a JSON-compatible per-agent record."""
    host = jax.tree.map(lambda value: np.asarray(value), sample)
    record = {
        "command_tracking_error": host.command_tracking_error.tolist(),
        "command_velocity": host.command_velocity.tolist(),
        "dataset_kind": dataset_kind,
        "done": bool(np.asarray(state.done)),
        "episode": episode,
        "episode_seed": episode_seed,
        "fallen": bool(host.fallen),
        "foot_contact": host.foot_contact.tolist(),
        "foot_position_world": host.foot_position_world.tolist(),
        "joint_position": host.joint_position.tolist(),
        "joint_position_target": host.joint_position_target.tolist(),
        "joint_velocity": host.joint_velocity.tolist(),
        "near_fall": bool(host.near_fall),
        "normalized_action": host.normalized_action.tolist(),
        "pelvis_angular_velocity": host.pelvis_angular_velocity.tolist(),
        "reward": float(np.asarray(state.reward)),
        "root_height": float(host.root_height),
        "root_linear_velocity_local": host.root_linear_velocity_local.tolist(),
        "root_linear_velocity_world": host.root_linear_velocity_world.tolist(),
        "root_orientation_wxyz": host.root_orientation_wxyz.tolist(),
        "root_position_world": host.root_position_world.tolist(),
        "schema_version": LOCOMOTION_DATASET_SCHEMA_VERSION,
        "timestep": timestep,
        "up_alignment": float(host.up_alignment),
    }
    if extra is not None:
        overlap = record.keys() & extra.keys()
        if overlap:
            raise ValueError(f"extra fields overlap base record: {sorted(overlap)}")
        record.update(extra)
    return record


def json_default(value):
    """Normalize common experiment types for the standard JSON encoder."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
