"""Versioned checkpoint I/O for the complete hierarchical learner state."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flax import serialization

from motionforge.training.hierarchical_update import (
    HierarchicalTrainState,
    HierarchicalUpdateConfig,
)

CHECKPOINT_SCHEMA = "motionforge.hierarchical_ppo"
CHECKPOINT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
STATE_FILENAME = "learner_state.msgpack"


def _state_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_hierarchical_checkpoint(
    path: Path,
    state: HierarchicalTrainState,
    *,
    update_config: HierarchicalUpdateConfig,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write one immutable hierarchical learner checkpoint."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")

    payload = serialization.to_bytes(state)
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "version": CHECKPOINT_VERSION,
        "state_file": STATE_FILENAME,
        "state_sha256": _state_digest(payload),
        "iteration": int(state.iteration),
        "strategy_updates": int(state.strategy_updates),
        "locomotion_updates": int(state.locomotion_updates),
        "update_config": asdict(update_config),
        "metadata": {} if metadata is None else dict(metadata),
    }
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        (temporary / STATE_FILENAME).write_bytes(payload)
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_hierarchical_checkpoint(
    path: Path,
    template: HierarchicalTrainState,
) -> tuple[HierarchicalTrainState, dict[str, Any]]:
    """Restore a checkpoint after validating its schema and state digest."""
    checkpoint = path.resolve()
    manifest_path = checkpoint / MANIFEST_FILENAME
    state_path = checkpoint / STATE_FILENAME
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not manifest_path.is_file() or not state_path.is_file():
        raise ValueError(f"incomplete hierarchical checkpoint: {checkpoint}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema: {manifest.get('schema')}")
    if manifest.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {manifest.get('version')}")
    if manifest.get("state_file") != STATE_FILENAME:
        raise ValueError("checkpoint manifest references an unexpected state file")

    payload = state_path.read_bytes()
    actual_digest = _state_digest(payload)
    if actual_digest != manifest.get("state_sha256"):
        raise ValueError("hierarchical checkpoint state digest does not match")
    restored = serialization.from_bytes(template, payload)
    counters = {
        "iteration": int(restored.iteration),
        "strategy_updates": int(restored.strategy_updates),
        "locomotion_updates": int(restored.locomotion_updates),
    }
    if any(counters[name] != manifest.get(name) for name in counters):
        raise ValueError("checkpoint counters do not match its manifest")
    return restored, manifest
