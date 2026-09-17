"""tag_observations.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Relative planar observations for the two-agent tag environment.

    Each agent observes the other agent's planar position and linear
    velocity in its own yaw-aligned heading frame. The module extracts
    physics state only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp

from motionforge.envs.two_g1 import TwoG1Model


@dataclass(frozen=True)
class AgentObservationLayout:
    root_qpos_start: int
    root_body_id: int
    global_velocity_sensor_slice: slice


@dataclass(frozen=True)
class TagObservationLayout:
    agents: tuple[
        AgentObservationLayout,
        AgentObservationLayout,
    ]


class RelativePlanarObservation(NamedTuple):
    position: jax.Array
    velocity: jax.Array


def build_tag_observation_layout(
    model_bundle: TwoG1Model,
) -> TagObservationLayout:
    """Resolve model addresses needed by runtime observation extraction."""
    layouts: list[AgentObservationLayout] = []

    for agent in model_bundle.agents:
        body_id = model_bundle.model.body(f"{agent.prefix}pelvis").id
        sensor_id = model_bundle.model.sensor(f"{agent.prefix}global_linvel_pelvis").id

        sensor_start = int(model_bundle.model.sensor_adr[sensor_id])
        sensor_dimension = int(model_bundle.model.sensor_dim[sensor_id])
        if sensor_dimension != 3:
            raise RuntimeError(
                f"{agent.prefix} global velocity sensor has "
                f"dimension {sensor_dimension}; expected 3"
            )

        layouts.append(
            AgentObservationLayout(
                root_qpos_start=agent.qpos_slice.start,
                root_body_id=body_id,
                global_velocity_sensor_slice=slice(
                    sensor_start,
                    sensor_start + sensor_dimension,
                ),
            )
        )

    return TagObservationLayout(agents=tuple(layouts))


def world_planar_to_heading(
    vector: jax.Array,
    root_quaternion: jax.Array,
) -> jax.Array:
    """Rotate a world-frame planar vector into the root yaw frame."""
    w, x, y, z = root_quaternion

    yaw = jp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    cosine = jp.cos(yaw)
    sine = jp.sin(yaw)

    return jp.asarray(
        [
            cosine * vector[0] + sine * vector[1],
            -sine * vector[0] + cosine * vector[1],
        ]
    )


def relative_planar_observation(
    data,
    layout: TagObservationLayout,
    observer_index: int,
) -> RelativePlanarObservation:
    """Observe the other agent in the observer's heading frame."""
    if observer_index not in (0, 1):
        raise ValueError("observer_index must be 0 or 1")

    other_index = 1 - observer_index
    observer = layout.agents[observer_index]
    other = layout.agents[other_index]

    observer_position = data.xpos[observer.root_body_id, :2]
    other_position = data.xpos[other.root_body_id, :2]

    observer_velocity = data.sensordata[observer.global_velocity_sensor_slice][:2]
    other_velocity = data.sensordata[other.global_velocity_sensor_slice][:2]

    quaternion_start = observer.root_qpos_start + 3
    observer_quaternion = data.qpos[quaternion_start : quaternion_start + 4]

    return RelativePlanarObservation(
        position=world_planar_to_heading(
            other_position - observer_position,
            observer_quaternion,
        ),
        velocity=world_planar_to_heading(
            other_velocity - observer_velocity,
            observer_quaternion,
        ),
    )
