"""Pure-standard-library selection of a fixed-reference calibration group."""

from __future__ import annotations

import math

COLLECTION_PROTOCOL = "tag_reference_calibration_collection_v1"


def select_calibration(
    collection,
    *,
    heading_noise_radians,
    stochastic_locomotion_actions,
    initial_separation,
    duration,
):
    """Select one matching protocol group; never substitute another distribution.

    Numeric comparison uses absolute tolerance 1e-9 and no relative tolerance.
    Only explicitly complete groups are usable. The producer owns validation of
    frozen-anchor provenance and required reference coverage. This helper does
    no filesystem, JAX, plotting, or network work.
    """
    if collection.get("protocol") != COLLECTION_PROTOCOL:
        raise ValueError("Unsupported reference calibration collection protocol")
    if not isinstance(stochastic_locomotion_actions, bool):
        raise TypeError("stochastic_locomotion_actions must be an explicit bool")
    requested = {
        "heading_noise_radians": heading_noise_radians,
        "initial_separation": initial_separation,
        "duration": duration,
    }
    if not all(math.isfinite(value) for value in requested.values()):
        raise ValueError("Calibration selection values must be finite")
    matches = []
    for group in collection.get("groups", []):
        if (
            group.get("stochastic_locomotion_actions")
            is not stochastic_locomotion_actions
        ):
            continue
        if all(
            isinstance(group.get(key), (int, float))
            and math.isfinite(group[key])
            and math.isclose(group[key], value, rel_tol=0, abs_tol=1e-9)
            for key, value in requested.items()
        ):
            matches.append(group)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one matching calibration group; found {len(matches)} for {requested}, motor_stochastic={stochastic_locomotion_actions}"
        )
    if matches[0].get("complete") is not True:
        raise ValueError(
            "Matching calibration group is incomplete; all four references are required"
        )
    if (
        "strategy_stochastic" in matches[0]
        and matches[0]["strategy_stochastic"] is not True
    ):
        raise ValueError("Calibration requires stochastic strategy actions")
    return matches[0]
