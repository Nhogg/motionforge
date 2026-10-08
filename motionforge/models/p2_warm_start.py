"""Warm-start the hierarchical locomotion module from an accepted P2 actor."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jp
from brax.training import checkpoint
from flax.core import freeze, unfreeze

from motionforge.compat.brax_checkpoint import load_ppo_network
from motionforge.models.hierarchical import (
    LOCOMOTION_ACTION_SIZE,
    LOCOMOTION_OBSERVATION_SIZE,
    LocomotionNormalization,
)

_P2_HIDDEN_LAYER_SIZES = (512, 256, 128)


def load_p2_checkpoint(checkpoint_path: Path):
    """Restore the saved P2 networks and full PPO parameter tuple."""
    resolved_path = checkpoint_path.resolve()
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"P2 checkpoint does not exist: {resolved_path}")
    config_path = resolved_path / "ppo_network_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"P2 network config does not exist: {config_path}")
    return load_ppo_network(config_path), checkpoint.load(resolved_path)


def p2_locomotion_normalization(processor_parameters: Any) -> LocomotionNormalization:
    """Extract the actor's 103-value normalization state."""
    mean = processor_parameters.mean
    std = processor_parameters.std
    if isinstance(mean, Mapping):
        mean = mean["state"]
        std = std["state"]
    mean = jp.asarray(mean)
    std = jp.asarray(std)
    if mean.shape != (LOCOMOTION_OBSERVATION_SIZE,):
        raise ValueError(f"P2 observation mean has unexpected shape {mean.shape}")
    if std.shape != (LOCOMOTION_OBSERVATION_SIZE,):
        raise ValueError(f"P2 observation std has unexpected shape {std.shape}")
    if bool(jp.any(std <= 0.0)):
        raise ValueError("P2 observation standard deviations must be positive")
    return LocomotionNormalization(mean=mean, std=std)


def migrate_p2_actor_parameters(
    hierarchical_parameters: Any,
    p2_policy_parameters: Any,
):
    """Copy the P2 deterministic actor into the locomotion parameter subtree.

    The P2 actor's final 58 outputs are the 29 action means followed by 29
    distribution-scale values. Deterministic inference uses only the means,
    so the first half initializes ``MotorHead.action``. The new locomotion
    value head is intentionally preserved because the P2 critic consumed a
    separate 216-value privileged observation.
    """
    target = unfreeze(hierarchical_parameters)
    source = p2_policy_parameters["params"]

    expected_source_shapes = (
        (LOCOMOTION_OBSERVATION_SIZE, _P2_HIDDEN_LAYER_SIZES[0]),
        (_P2_HIDDEN_LAYER_SIZES[0], _P2_HIDDEN_LAYER_SIZES[1]),
        (_P2_HIDDEN_LAYER_SIZES[1], _P2_HIDDEN_LAYER_SIZES[2]),
    )
    for index, expected_kernel_shape in enumerate(expected_source_shapes):
        source_layer = source[f"hidden_{index}"]
        target_layer = target["locomotion"]["encoder"][f"Dense_{index}"]
        if tuple(source_layer["kernel"].shape) != expected_kernel_shape:
            raise ValueError(
                f"P2 hidden_{index} kernel has shape "
                f"{source_layer['kernel'].shape}; expected {expected_kernel_shape}"
            )
        if source_layer["kernel"].shape != target_layer["kernel"].shape:
            raise ValueError(
                f"P2 hidden_{index} does not match hierarchical locomotion encoder"
            )
        target_layer["kernel"] = jp.asarray(source_layer["kernel"])
        target_layer["bias"] = jp.asarray(source_layer["bias"])

    source_output = source["hidden_3"]
    expected_output_size = 2 * LOCOMOTION_ACTION_SIZE
    if source_output["kernel"].shape != (
        _P2_HIDDEN_LAYER_SIZES[-1],
        expected_output_size,
    ):
        raise ValueError(
            f"P2 output kernel has unexpected shape {source_output['kernel'].shape}"
        )
    action_head = target["locomotion"]["motor_head"]["action"]
    action_head["kernel"] = jp.asarray(
        source_output["kernel"][:, :LOCOMOTION_ACTION_SIZE]
    )
    action_head["bias"] = jp.asarray(
        source_output["bias"][:LOCOMOTION_ACTION_SIZE]
    )
    return freeze(target)
