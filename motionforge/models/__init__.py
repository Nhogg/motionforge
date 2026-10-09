"""Trainable model architectures used by MotionForge."""

from motionforge.models.distributions import (
    diagonal_normal_entropy,
    tanh_normal_log_prob,
    tanh_normal_sample_and_log_prob,
)
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
    StrategyModule,
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
    "StrategyModule",
    "StrategyObservation",
    "StrategyOutput",
    "diagonal_normal_entropy",
    "load_p2_checkpoint",
    "migrate_p2_actor_parameters",
    "p2_locomotion_normalization",
    "tanh_normal_log_prob",
    "tanh_normal_sample_and_log_prob",
]
