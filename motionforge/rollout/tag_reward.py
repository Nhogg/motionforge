"""Strategy-only TAG shaping, evaluated once per strategy decision interval."""

import math
from dataclasses import dataclass, field

import jax.numpy as jp
from flax import struct


@dataclass(unsafe_hash=True)
class TagProgressConfig:
    enabled: bool = True
    alpha: float = 0.25
    distance_normalization_m: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("potential alpha must be finite and nonnegative")
        if self.distance_normalization_m is not None and (
            not math.isfinite(self.distance_normalization_m)
            or self.distance_normalization_m <= 0
        ):
            raise ValueError(
                "potential distance normalization must be finite and positive"
            )

    def normalization_m(self, arena_half_extent):
        """Square arena diameter is its fixed corner-to-corner diagonal."""
        return self.distance_normalization_m or (
            2.0 * math.sqrt(2.0) * arena_half_extent
        )


@dataclass(unsafe_hash=True)
class TagRewardConfig:
    progress: TagProgressConfig = field(default_factory=TagProgressConfig)
    command_smoothness_weight: float = 0.005
    own_fall_penalty: float = 1.0
    own_oob_penalty: float = 1.0
    boundary_weight: float = 0.005
    boundary_onset_m: float = 1.0
    boundary_max_penalty: float = 0.02

    def __post_init__(self):
        for name in (
            "command_smoothness_weight",
            "own_fall_penalty",
            "own_oob_penalty",
            "boundary_weight",
            "boundary_max_penalty",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.boundary_onset_m) or self.boundary_onset_m <= 0:
            raise ValueError("boundary onset must be finite and positive")


def tag_reward_config_from_dict(payload):
    """Restore a v2 reward config; never reinterpret old clipped settings."""
    values = dict(payload)
    progress = values.pop("progress", {})
    if "weight" in progress or "distance_scale_m" in progress:
        raise ValueError(
            "checkpoint uses legacy clipped progress shaping, not potential shaping"
        )
    return TagRewardConfig(progress=TagProgressConfig(**progress), **values)


@struct.dataclass
class TagRewardTerms:
    progress: object
    tag: object
    command_smoothness: object
    boundary: object
    fall: object
    oob: object
    total: object


def tag_strategy_reward_terms(
    *,
    previous_distance,
    current_distance,
    command,
    previous_command,
    position,
    arena_half_extent,
    tagged,
    own_fall,
    own_oob,
    learner_is_pursuer,
    config,
    discount=1.0,
):
    """No proximity bonus and no timeout outcome reward. Tag component is ±1."""
    normalization = config.progress.normalization_m(arena_half_extent)
    potential_previous = -config.progress.alpha * previous_distance / normalization
    potential_current = -config.progress.alpha * current_distance / normalization
    progress = (
        discount * potential_current - potential_previous
        if config.progress.enabled
        else jp.zeros_like(current_distance)
    )
    progress = jp.where(learner_is_pursuer, progress, -progress)
    tag = jp.where(learner_is_pursuer, 1.0, -1.0) * tagged.astype(jp.float32)
    smoothness = -config.command_smoothness_weight * jp.sum(
        jp.square(command - previous_command), axis=-1
    )
    clearance = arena_half_extent - jp.max(jp.abs(position), axis=-1)
    boundary = -jp.minimum(
        config.boundary_max_penalty,
        config.boundary_weight
        * jp.square(jp.clip(1.0 - clearance / config.boundary_onset_m, 0.0, 1.0)),
    )
    fall = -config.own_fall_penalty * own_fall.astype(jp.float32)
    oob = -config.own_oob_penalty * own_oob.astype(jp.float32)
    total = progress + tag + smoothness + boundary + fall + oob
    return TagRewardTerms(progress, tag, smoothness, boundary, fall, oob, total)
