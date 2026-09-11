"""Exercise independent fall classification for both agents."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from motionforge.cli import run_hydra
from motionforge.envs.tag_fall import (
    FallDetectionConfig,
    build_fall_detection_layout,
    detect_falls,
)
from motionforge.envs.two_g1 import (
    build_two_g1_model,
    make_two_g1_data,
)


@dataclass
class Config:
    seed: int = 0
    separation: float = 2.0
    minimum_root_height: float = 0.45
    minimum_up_alignment: float = 0.50
    output: Path = Path("logs/p3/tag_fall_a.json")


def main(config: Config) -> None:
    bundle = build_two_g1_model()
    layout = build_fall_detection_layout(bundle)
    fall_config = FallDetectionConfig(
        minimum_root_height=config.minimum_root_height,
        minimum_up_alignment=config.minimum_up_alignment,
    )

    initial_data = make_two_g1_data(
        bundle,
        separation=config.separation,
    )
    initial = detect_falls(initial_data, layout, fall_config)

    low_data = make_two_g1_data(
        bundle,
        separation=config.separation,
    )
    low_data.qpos[bundle.agents[0].qpos_slice.start + 2] = (
        config.minimum_root_height - 0.01
    )
    low = detect_falls(low_data, layout, fall_config)

    tilted_data = make_two_g1_data(
        bundle,
        separation=config.separation,
    )
    half_angle = np.pi / 4.0
    quaternion_start = bundle.agents[1].qpos_slice.start + 3
    tilted_data.qpos[quaternion_start : quaternion_start + 4] = np.array(
        [
            np.cos(half_angle),
            np.sin(half_angle),
            0.0,
            0.0,
        ]
    )
    tilted = detect_falls(tilted_data, layout, fall_config)

    initial_fallen = np.asarray(initial.fallen, dtype=bool)
    low_fallen = np.asarray(low.fallen, dtype=bool)
    tilted_fallen = np.asarray(tilted.fallen, dtype=bool)

    diagnostic_arrays = (
        np.asarray(initial.root_height),
        np.asarray(initial.up_alignment),
        np.asarray(low.root_height),
        np.asarray(low.up_alignment),
        np.asarray(tilted.root_height),
        np.asarray(tilted.up_alignment),
    )

    checks = {
        "diagnostics_finite": bool(
            all(np.isfinite(values).all() for values in diagnostic_arrays)
        ),
        "initial_agents_upright": bool(not initial_fallen.any()),
        "initial_state_finite": bool(
            np.asarray(initial.state_finite, dtype=bool).all()
        ),
        "low_agent_detected": bool(low_fallen[0]),
        "low_fixture_isolated": bool(not low_fallen[1]),
        "tilted_agent_detected": bool(tilted_fallen[1]),
        "tilted_fixture_isolated": bool(not tilted_fallen[0]),
    }

    result = {
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_fall_detection",
        "fixtures": {
            "initial": {
                "fallen": initial_fallen.tolist(),
                "root_height": np.asarray(initial.root_height).tolist(),
                "up_alignment": np.asarray(initial.up_alignment).tolist(),
            },
            "low_agent0": {
                "fallen": low_fallen.tolist(),
                "root_height": np.asarray(low.root_height).tolist(),
                "up_alignment": np.asarray(low.up_alignment).tolist(),
            },
            "tilted_agent1": {
                "fallen": tilted_fallen.tolist(),
                "root_height": np.asarray(tilted.root_height).tolist(),
                "up_alignment": np.asarray(tilted.up_alignment).tolist(),
            },
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
