"""test_tag_roles.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

from motionforge.envs.tag_roles import (
    TagRole,
    assign_tag_roles,
)
from motionforge.envs.two_g1 import build_two_g1_model


@dataclass(frozen=True)
class Config:
    seed: int = 0
    pursuer_index: int = 0
    output: Path = Path("logs/p3/tag_roles_a.json")


def main(config: Config) -> None:
    bundle = build_two_g1_model()
    assignment = assign_tag_roles(
        bundle.agents,
        pursuer_index=config.pursuer_index,
    )

    pursuer = assignment.by_role(TagRole.PURSUER)
    evader = assignment.by_role(TagRole.EVADER)

    checks = {
        "distinct_agents": pursuer.index != evader.index,
        "evader_assigned": evader.role == TagRole.EVADER,
        "model_prefixes_match": (
            assignment.agents[0].model_prefix == bundle.agents[0].prefix
            and assignment.agents[1].model_prefix == bundle.agents[1].prefix
        ),
        "pursuer_assigned": pursuer.role == TagRole.PURSUER,
        "pursuer_index": pursuer.index == config.pursuer_index,
        "role_count": len(assignment.agents) == 2,
        "roles_unique": len({agent.role for agent in assignment.agents}) == 2,
    }

    result = {
        "agents": [
            {
                "index": agent.index,
                "model_prefix": agent.model_prefix,
                "role": agent.role.value,
            }
            for agent in assignment.agents
        ],
        "checks": checks,
        "config": {
            **asdict(config),
            "output": str(config.output),
        },
        "experiment": "two_g1_tag_role_assignment",
        "passed": all(checks.values()),
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
    main(tyro.cli(Config))
