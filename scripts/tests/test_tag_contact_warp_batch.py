"""Regression for world-isolated contact labels in batched Warp states."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from motionforge.cli import run_hydra
from motionforge.envs import MultiAgentTaskConfig, MultiAgentTaskEnv
from motionforge.envs.tag_contact import detect_tag_contact
from motionforge.league import FixedLeague, LeagueOpponent


@dataclass
class Config:
    seed: int = 23
    naconmax: int = 64
    njmax: int = 256
    output: Path = Path("logs/p3/tag_contact_warp_batch_a.json")


def main(config: Config) -> None:
    environment = MultiAgentTaskEnv(
        league=FixedLeague(LeagueOpponent("test", "scripted", None)),
        config=MultiAgentTaskConfig(
            naconmax=config.naconmax,
            njmax=config.njmax,
        ),
    )
    agent0, agent1 = environment.model_bundle.agents
    world_indices = jp.arange(2, dtype=jp.int32)
    initial = jax.vmap(environment.reset)(
        jax.random.split(jax.random.PRNGKey(config.seed), 2),
        world_indices,
    )
    base_qpos = initial.physics.data.qpos

    def detect(touching: tuple[int, ...]) -> tuple[list[bool], list[int]]:
        qpos = base_qpos
        for world in touching:
            qpos = qpos.at[
                world,
                agent1.qpos_slice.start : agent1.qpos_slice.start + 3,
            ].set(
                qpos[
                    world,
                    agent0.qpos_slice.start : agent0.qpos_slice.start + 3,
                ]
            )
        data = mjx.forward(environment.model, initial.physics.data.replace(qpos=qpos))
        contact = detect_tag_contact(
            data,
            environment.contact_layout,
            world_index=world_indices,
        )
        return (
            np.asarray(contact.occurred).tolist(),
            np.asarray(contact.count).tolist(),
        )

    cases = {
        "separated": detect(()),
        "world0_touching": detect((0,)),
        "world1_touching": detect((1,)),
        "both_touching": detect((0, 1)),
    }
    checks = {
        "separated": cases["separated"] == ([False, False], [0, 0]),
        "world0_isolated": cases["world0_touching"][0] == [True, False],
        "world1_isolated": cases["world1_touching"][0] == [False, True],
        "both_worlds_detected": cases["both_touching"][0] == [True, True],
        "world0_count_positive": cases["world0_touching"][1][0] > 0,
        "world1_count_positive": cases["world1_touching"][1][1] > 0,
        "world_indices_preserved": np.array_equal(
            np.asarray(initial.physics.world_index),
            np.asarray([0, 1]),
        ),
    }
    result = {
        "backend": jax.default_backend(),
        "cases": cases,
        "checks": checks,
        "config": {**asdict(config), "output": str(config.output)},
        "experiment": "batched_warp_tag_contact_isolation",
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
