"""Small JAX-native action distributions used by hierarchical PPO."""

from __future__ import annotations

import math

import jax
import jax.numpy as jp


def tanh_normal_sample_and_log_prob(
    key: jax.Array,
    location: jax.Array,
    log_std: jax.Array,
    scale: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Draw a reparameterized bounded action and return its log density."""
    noise = jax.random.normal(key, location.shape)
    latent = location + jp.exp(log_std) * noise
    unit_action = jp.tanh(latent)
    action = unit_action * scale
    log_prob = -0.5 * (
        jp.square((latent - location) / jp.exp(log_std))
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    log_jacobian = jp.log(scale) + jp.log(
        jp.maximum(1.0 - jp.square(unit_action), 1e-6)
    )
    return action, jp.sum(log_prob - log_jacobian, axis=-1)


def tanh_normal_log_prob(
    action: jax.Array,
    location: jax.Array,
    log_std: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Evaluate a scaled tanh-normal density at a bounded action."""
    unit_action = jp.clip(action / scale, -1.0 + 1e-6, 1.0 - 1e-6)
    latent = jp.arctanh(unit_action)
    log_prob = -0.5 * (
        jp.square((latent - location) / jp.exp(log_std))
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    log_jacobian = jp.log(scale) + jp.log(
        jp.maximum(1.0 - jp.square(unit_action), 1e-6)
    )
    return jp.sum(log_prob - log_jacobian, axis=-1)


def diagonal_normal_entropy(log_std: jax.Array) -> jax.Array:
    """Return the base diagonal-normal entropy used for PPO regularization."""
    return jp.sum(
        log_std + 0.5 * (1.0 + math.log(2.0 * math.pi)), axis=-1
    )
