"""Exercise independent OOB classification for both tag agents."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs import (
    BoundsDetectionConfig,
    BoundsObservation,
    build_bounds_detection_layout,
    build_two_g1_model,
    detect_out_of_bounds,
    make_two_g1_data,
)


@dataclass
class Config:
    seed: int = 0
    separation: float = 2.0
    arena_half_extent: float = 4.0
    output: Path = Path("logs/p3/tag_bounds_a.json")


def json_numbers(value: Any) -> Any:
    """Convert arrays to JSON values while making non-finite values explicit."""
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, float) and not np.isfinite(scalar):
            return str(scalar)
        return scalar
    return [json_numbers(item) for item in array]


def observation_result(observation: BoundsObservation) -> dict[str, Any]:
    return {
        "boundary_excess": json_numbers(observation.boundary_excess),
        "out_of_bounds": np.asarray(
            observation.out_of_bounds,
            dtype=bool,
        ).tolist(),
        "planar_position": json_numbers(observation.planar_position),
        "state_finite": np.asarray(
            observation.state_finite,
            dtype=bool,
        ).tolist(),
    }


def main(config: Config) -> None:
    bundle = build_two_g1_model()
    layout = build_bounds_detection_layout(bundle)
    bounds_config = BoundsDetectionConfig(
        arena_half_extent=config.arena_half_extent,
    )

    initial_data = make_two_g1_data(bundle, separation=config.separation)
    initial = detect_out_of_bounds(initial_data, layout, bounds_config)

    boundary_data = make_two_g1_data(bundle, separation=config.separation)
    boundary_data.qpos[bundle.agents[0].qpos_slice.start] = (
        config.arena_half_extent
    )
    boundary = detect_out_of_bounds(boundary_data, layout, bounds_config)

    x_outside_data = make_two_g1_data(bundle, separation=config.separation)
    x_outside_data.qpos[bundle.agents[0].qpos_slice.start] = (
        config.arena_half_extent + 0.01
    )
    x_outside = detect_out_of_bounds(x_outside_data, layout, bounds_config)

    y_outside_data = make_two_g1_data(bundle, separation=config.separation)
    y_outside_data.qpos[bundle.agents[1].qpos_slice.start + 1] = (
        -config.arena_half_extent - 0.01
    )
    y_outside = detect_out_of_bounds(y_outside_data, layout, bounds_config)

    nonfinite_data = make_two_g1_data(bundle, separation=config.separation)
    nonfinite_data.qpos[bundle.agents[0].qpos_slice.start] = np.nan
    nonfinite = detect_out_of_bounds(nonfinite_data, layout, bounds_config)

    initial_oob = np.asarray(initial.out_of_bounds, dtype=bool)
    boundary_oob = np.asarray(boundary.out_of_bounds, dtype=bool)
    x_outside_oob = np.asarray(x_outside.out_of_bounds, dtype=bool)
    y_outside_oob = np.asarray(y_outside.out_of_bounds, dtype=bool)
    nonfinite_oob = np.asarray(nonfinite.out_of_bounds, dtype=bool)

    finite_fixture_diagnostics = (
        np.asarray(initial.planar_position),
        np.asarray(initial.boundary_excess),
        np.asarray(boundary.planar_position),
        np.asarray(boundary.boundary_excess),
        np.asarray(x_outside.planar_position),
        np.asarray(x_outside.boundary_excess),
        np.asarray(y_outside.planar_position),
        np.asarray(y_outside.boundary_excess),
    )

    checks = {
        "boundary_is_inclusive": bool(not boundary_oob[0]),
        "finite_fixture_diagnostics": bool(
            all(np.isfinite(values).all() for values in finite_fixture_diagnostics)
        ),
        "initial_agents_inside": bool(not initial_oob.any()),
        "initial_state_finite": bool(
            np.asarray(initial.state_finite, dtype=bool).all()
        ),
        "nonfinite_state_detected": bool(nonfinite_oob[0]),
        "x_fixture_isolated": bool(not x_outside_oob[1]),
        "x_outside_detected": bool(x_outside_oob[0]),
        "y_fixture_isolated": bool(not y_outside_oob[0]),
        "y_outside_detected": bool(y_outside_oob[1]),
    }

    result = {
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_out_of_bounds_detection",
        "fixtures": {
            "boundary_agent0": observation_result(boundary),
            "initial": observation_result(initial),
            "nonfinite_agent0": observation_result(nonfinite),
            "x_outside_agent0": observation_result(x_outside),
            "y_outside_agent1": observation_result(y_outside),
        },
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "seed": config.seed,
    }

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run_hydra(Config, main)
