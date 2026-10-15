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
import jax.numpy as jp

from motionforge.cli import run_hydra
from motionforge.envs.tag_contact import (
    build_tag_contact_layout,
    detect_tag_contact,
)
from motionforge.envs.two_g1 import (
    build_two_g1_model,
    make_two_g1_data,
)


@dataclass
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

    class WarpImpl:
        def __init__(self, world_ids, active_count):
            pair = [
                layout.agent0_geom_ids[0],
                layout.agent1_geom_ids[0],
            ]
            self.contact__geom = jp.asarray([pair] * len(world_ids), dtype=jp.int32)
            self.contact__dist = jp.asarray(
                [-0.01] * len(world_ids),
                dtype=jp.float32,
            )
            self.contact__worldid = jp.asarray(world_ids, dtype=jp.int32)
            self.nacon = jp.asarray([active_count], dtype=jp.int32)
            self.naconmax = len(world_ids)

    class WarpData:
        def __init__(self, world_ids, active_count):
            self._impl = WarpImpl(world_ids, active_count)

    def isolated_worlds(world_ids, active_count):
        data = WarpData(world_ids, active_count)
        return tuple(
            detect_tag_contact(data, layout, world_index=index)
            for index in (0, 1)
        )

    world0_only = isolated_worlds([0, 1, 0, 1], 1)
    world1_only = isolated_worlds([1, 0, 1, 0], 1)
    both_worlds = isolated_worlds([0, 1, 0, 1], 2)
    reversed_worlds = isolated_worlds([1, 0, 1, 0], 2)
    no_worlds = isolated_worlds([0, 1, 0, 1], 0)

    checks = {
        "inter_agent_pairs_created": bundle.model.npair == 74,
        "no_tag_when_separated": not separated_occurred,
        "tag_when_overlapping": overlapping_occurred,
        "tag_contact_count": overlapping_count > 0,
        "tag_distance_finite": bool(np.isfinite(overlapping_minimum_distance)),
        "tag_distance_nonpositive": bool(overlapping_minimum_distance <= 0.0),
        "warp_world0_only_isolated": (
            bool(np.asarray(world0_only[0].occurred))
            and not bool(np.asarray(world0_only[1].occurred))
            and int(np.asarray(world0_only[0].count)) == 1
            and int(np.asarray(world0_only[1].count)) == 0
        ),
        "warp_world1_only_isolated": (
            not bool(np.asarray(world1_only[0].occurred))
            and bool(np.asarray(world1_only[1].occurred))
            and int(np.asarray(world1_only[0].count)) == 0
            and int(np.asarray(world1_only[1].count)) == 1
        ),
        "warp_both_worlds_detected": all(
            bool(np.asarray(observation.occurred))
            and int(np.asarray(observation.count)) == 1
            for observation in both_worlds
        ),
        "warp_reversed_world_order_detected": all(
            bool(np.asarray(observation.occurred))
            and int(np.asarray(observation.count)) == 1
            for observation in reversed_worlds
        ),
        "warp_no_active_contacts": all(
            not bool(np.asarray(observation.occurred))
            and int(np.asarray(observation.count)) == 0
            for observation in no_worlds
        ),
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
    run_hydra(Config, main)
