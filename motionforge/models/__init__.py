"""Trainable model architectures used by MotionForge."""

from motionforge.models.hierarchical import (
    LOCOMOTION_ACTION_SIZE,
    LOCOMOTION_OBSERVATION_SIZE,
    STRATEGY_ACTION_SIZE,
    STRATEGY_OBSERVATION_SIZE,
    HierarchicalPolicy,
    HierarchicalPolicyConfig,
    HierarchicalPolicyOutput,
    LocomotionModule,
    LocomotionNormalization,
    LocomotionObservation,
    LocomotionOutput,
    StrategyObservation,
    StrategyOutput,
)
from motionforge.models.p2_warm_start import (
    load_p2_checkpoint,
    migrate_p2_actor_parameters,
    p2_locomotion_normalization,
)

__all__ = [
    "LOCOMOTION_ACTION_SIZE",
    "LOCOMOTION_OBSERVATION_SIZE",
    "STRATEGY_ACTION_SIZE",
    "STRATEGY_OBSERVATION_SIZE",
    "HierarchicalPolicy",
    "HierarchicalPolicyConfig",
    "HierarchicalPolicyOutput",
    "LocomotionModule",
    "LocomotionNormalization",
    "LocomotionObservation",
    "LocomotionOutput",
    "StrategyObservation",
    "StrategyOutput",
    "load_p2_checkpoint",
    "migrate_p2_actor_parameters",
    "p2_locomotion_normalization",
]
