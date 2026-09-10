"""tag_roles.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Role metadata for a two-agent tag environment.

    Roles are assigned to stable model-agent indices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from motionforge.envs.two_g1 import G1ModelLayout


class TagRole(StrEnum):
    PURSUER = "pursuer"
    EVADER = "evader"


@dataclass(frozen=True)
class TagAgent:
    index: int
    model_prefix: str
    role: TagRole


@dataclass(frozen=True)
class TagRoleAssignment:
    agents: tuple[TagAgent, TagAgent]

    def __post_init__(self) -> None:
        if len(self.agents) != 2:
            raise ValueError("Tag requires exactly two agents")

        indices = {agent.index for agent in self.agents}
        prefixes = {agent.model_prefix for agent in self.agents}
        roles = {agent.role for agent in self.agents}

        if indices != {0, 1}:
            raise ValueError(f"Agent indices must be 0 and 1; got {indices}")
        if len(prefixes) != 2:
            raise ValueError("Agent model prefixes must be distinct")
        if roles != {TagRole.PURSUER, TagRole.EVADER}:
            raise ValueError("Exactly one pursuer and one evader are required")

    def by_role(self, role: TagRole) -> TagAgent:
        """Return the uniquely assigned agent for a role."""
        for agent in self.agents:
            if agent.role == role:
                return agent
        raise RuntimeError(f"Role is not assigned: {role}")


def assign_tag_roles(
    layouts: Sequence[G1ModelLayout],
    pursuer_index: int = 0,
) -> TagRoleAssignment:
    """Assign pursuer and evader roles to two model layouts."""
    if len(layouts) != 2:
        raise ValueError(f"Tag requires exactly two model layouts; got {len(layouts)}")
    if pursuer_index not in (0, 1):
        raise ValueError("--pursuer-index must be 0 or 1")

    agents = tuple(
        TagAgent(
            index=index,
            model_prefix=layout.prefix,
            role=(TagRole.PURSUER if index == pursuer_index else TagRole.EVADER),
        )
        for index, layout in enumerate(layouts)
    )

    return TagRoleAssignment(agents=agents)
