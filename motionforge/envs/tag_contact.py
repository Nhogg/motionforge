"""tag_contact.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Inter-agent contact detection for the two-G1 tag environment.

    The detector classifies contacts using precomputed geom-id ownership.
    Floor contacts and contacts within one robot are excluded. It returns
    contact presence, count, and minimum signed contact distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp

from motionforge.envs.two_g1 import (
    G1_COLLISION_GEOM_NAMES,
    TwoG1Model,
)


@dataclass(frozen=True)
class TagContactLayout:
    agent0_geom_ids: tuple[int, ...]
    agent1_geom_ids: tuple[int, ...]


class TagContactObservation(NamedTuple):
    occurred: jax.Array
    count: jax.Array
    minimum_distance: jax.Array


def build_tag_contact_layout(
    model_bundle: TwoG1Model,
) -> TagContactLayout:
    """Resolve the collision geom IDs belonging to each agent."""
    agent_geom_ids: list[tuple[int, ...]] = []

    for agent in model_bundle.agents:
        geom_ids = tuple(
            model_bundle.model.geom(f"{agent.prefix}{geom_name}").id
            for geom_name in G1_COLLISION_GEOM_NAMES
        )
        agent_geom_ids.append(geom_ids)

    return TagContactLayout(
        agent0_geom_ids=agent_geom_ids[0],
        agent1_geom_ids=agent_geom_ids[1],
    )


def _belongs_to(
    geom_ids: jax.Array,
    candidates: tuple[int, ...],
) -> jax.Array:
    candidate_array = jp.asarray(candidates)
    return jp.any(
        geom_ids[..., None] == candidate_array,
        axis=-1,
    )


def _contact_buffers(data) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return contact geom pairs, distances, and an active-contact mask."""
    if hasattr(data, "contact"):
        contact_geoms = jp.asarray(data.contact.geom)
        distances = jp.asarray(data.contact.dist)
        active = jp.ones(distances.shape, dtype=bool)
        return contact_geoms, distances, active

    impl = data._impl
    contact_geoms = jp.asarray(impl.contact__geom)
    distances = jp.asarray(impl.contact__dist)
    world_ids = jp.asarray(impl.contact__worldid)
    active_counts = jp.asarray(impl.nacon)

    slots = jp.arange(distances.shape[0]) % impl.naconmax
    active = slots < active_counts[world_ids]

    return contact_geoms, distances, active


def detect_tag_contact(
    data,
    layout: TagContactLayout,
) -> TagContactObservation:
    """Detect active contacts containing one geom from each agent."""
    contact_geoms, distances, active = _contact_buffers(data)
    geom1 = contact_geoms[..., 0]
    geom2 = contact_geoms[..., 1]

    agent0_geom1 = _belongs_to(geom1, layout.agent0_geom_ids)
    agent0_geom2 = _belongs_to(geom2, layout.agent0_geom_ids)
    agent1_geom1 = _belongs_to(geom1, layout.agent1_geom_ids)
    agent1_geom2 = _belongs_to(geom2, layout.agent1_geom_ids)

    inter_agent_contact = active & (
        (agent0_geom1 & agent1_geom2) | (agent1_geom1 & agent0_geom2)
    )

    return TagContactObservation(
        occurred=jp.any(inter_agent_contact),
        count=jp.sum(inter_agent_contact),
        minimum_distance=jp.min(jp.where(inter_agent_contact, distances, jp.inf)),
    )
