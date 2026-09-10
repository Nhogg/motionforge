"""test_tag_contact.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

from motionforge.envs.tag_contact import (
    build_tag_contact_layout,
    detect_tag_contact,
)
from motionforge.envs.two_g1 import (
    build_two_g1_model,
    make_two_g1_data,
)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    separated_distance: float = 2.0
    output: Path = Path("logs/p3/tag_contact_a.json")


def main(config: Config) -> None:
    bundle = build_two_g1_model(
        enable_inter_agent_collision=True,
    )
    layout = build_tag_contact_layout(bundle)

    separated_data = make_two_g1_data(
        bundle,
        separation=config.separated_distance,
    )
    separated_contact = detect_tag_contact(
        separated_data,
        layout,
    )

    overlapping_data = make_two_g1_data(
        bundle,
        separation=config.separated_distance,
    )

    agent0 = bundle.agents[0]
    agent1 = bundle.agents[1]

    agent0_root = overlapping_data.qpos[
        agent0.qpos_slice.start : agent0.qpos_slice.start + 3
    ].copy()
    overlapping_data.qpos[agent1.qpos_slice.start : agent1.qpos_slice.start + 3] = (
        agent0_root
    )

    mujoco.mj_forward(bundle.model, overlapping_data)
    overlapping_contact = detect_tag_contact(
        overlapping_data,
        layout,
    )

    separated_occurred = bool(np.asarray(separated_contact.occurred))
    overlapping_occurred = bool(np.asarray(overlapping_contact.occurred))
    overlapping_count = int(np.asarray(overlapping_contact.count))
    overlapping_minimum_distance = float(
        np.asarray(overlapping_contact.minimum_distance)
    )

    checks = {
        "inter_agent_pairs_created": bundle.model.npair == 74,
        "no_tag_when_separated": not separated_occurred,
        "tag_when_overlapping": overlapping_occurred,
        "tag_contact_count": overlapping_count > 0,
        "tag_distance_finite": bool(np.isfinite(overlapping_minimum_distance)),
        "tag_distance_nonpositive": bool(overlapping_minimum_distance <= 0.0),
    }

    result = {
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_tag_contact_detection",
        "inter_agent_pair_count": bundle.model.npair - 10,
        "model_pair_count": bundle.model.npair,
        "overlapping": {
            "contact_count": overlapping_count,
            "minimum_distance": overlapping_minimum_distance,
            "occurred": overlapping_occurred,
        },
        "passed": bool(all(checks.values())),
        "python_version": platform.python_version(),
        "seed": config.seed,
        "separated": {
            "contact_count": int(np.asarray(separated_contact.count)),
            "minimum_distance": (
                float(np.asarray(separated_contact.minimum_distance))
                if separated_occurred
                else None
            ),
            "occurred": separated_occurred,
        },
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
    main(tyro.cli(Config))
