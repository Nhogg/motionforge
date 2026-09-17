"""Adapt shared two-G1 physics state to the P2 locomotion policy interface.

The adapter constructs each agent's 103-element actor observation from the
combined tag model. It owns no policy parameters, commands, or physics steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp

from motionforge.envs.two_g1 import TwoG1Model


@dataclass(frozen=True)
class G1TagPolicyAgentLayout:
    qpos_slice: slice
    qvel_slice: slice
    local_velocity_sensor_slice: slice
    gyro_sensor_slice: slice
    pelvis_imu_site_id: int


@dataclass(frozen=True)
class G1TagPolicyObservationLayout:
    agents: tuple[G1TagPolicyAgentLayout, G1TagPolicyAgentLayout]


def _sensor_slice(model, name: str, expected_dimension: int) -> slice:
    sensor_id = model.sensor(name).id
    start = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])
    if dimension != expected_dimension:
        raise RuntimeError(
            f"Sensor {name!r} has dimension {dimension}; expected {expected_dimension}"
        )
    return slice(start, start + dimension)


def build_g1_tag_policy_observation_layout(
    model_bundle: TwoG1Model,
) -> G1TagPolicyObservationLayout:
    """Resolve combined-model addresses needed by each locomotion actor."""
    layouts = []
    for agent in model_bundle.agents:
        layouts.append(
            G1TagPolicyAgentLayout(
                qpos_slice=agent.qpos_slice,
                qvel_slice=agent.qvel_slice,
                local_velocity_sensor_slice=_sensor_slice(
                    model_bundle.model,
                    f"{agent.prefix}local_linvel_pelvis",
                    3,
                ),
                gyro_sensor_slice=_sensor_slice(
                    model_bundle.model,
                    f"{agent.prefix}gyro_pelvis",
                    3,
                ),
                pelvis_imu_site_id=model_bundle.model.site(
                    f"{agent.prefix}imu_in_pelvis"
                ).id,
            )
        )
    return G1TagPolicyObservationLayout(agents=tuple(layouts))


def g1_tag_policy_observation(
    data,
    layout: G1TagPolicyObservationLayout,
    agent_index: int,
    command: jax.Array,
    last_action: jax.Array,
    phase: jax.Array,
    default_pose: jax.Array,
) -> dict[str, jax.Array]:
    """Construct one actor observation matching the accepted P2 checkpoint."""
    if agent_index not in (0, 1):
        raise ValueError("agent_index must be 0 or 1")
    if command.shape != (3,):
        raise ValueError(f"command must have shape (3,), got {command.shape}")
    if last_action.shape != (29,):
        raise ValueError(f"last_action must have shape (29,), got {last_action.shape}")
    if phase.shape != (2,):
        raise ValueError(f"phase must have shape (2,), got {phase.shape}")
    if default_pose.shape != (29,):
        raise ValueError(
            f"default_pose must have shape (29,), got {default_pose.shape}"
        )

    agent = layout.agents[agent_index]
    joint_position = data.qpos[agent.qpos_slice.start + 7 : agent.qpos_slice.stop]
    joint_velocity = data.qvel[agent.qvel_slice.start + 6 : agent.qvel_slice.stop]
    gravity = data.site_xmat[agent.pelvis_imu_site_id].T @ jp.asarray([0.0, 0.0, -1.0])

    state = jp.hstack(
        [
            data.sensordata[agent.local_velocity_sensor_slice],
            data.sensordata[agent.gyro_sensor_slice],
            gravity,
            command,
            joint_position - default_pose,
            joint_velocity,
            last_action,
            jp.cos(phase),
            jp.sin(phase),
        ]
    )
    return {"state": state}
