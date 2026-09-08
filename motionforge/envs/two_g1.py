"""two_g1.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Compose two namespace-isolated Unitree G1 robots on one flat arena.

    Model reuses MuJoCo Playground's pinned G1 MJCF and Menagerie assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
from mujoco_playground._src.locomotion.g1 import base, g1_constants


@dataclass(frozen=True)
class G1ModelLayout:
    prefix: str
    qpos_slice: slice
    qvel_slice: slice
    actuator_ids: tuple[int, ...]


@dataclass(frozen=True)
class TwoG1Model:
    model: mujoco.MjModel
    agents: tuple[G1ModelLayout, G1ModelLayout]


def build_two_g1_model(timestep: float = 0.002) -> TwoG1Model:
    """Compile two prefixed G1 models and one shared flat floor."""
    if timestep <= 0.0:
        raise ValueError("timestep must be positive")

    assets = base.get_assets()
    robot_path = str(g1_constants.ROOT_PATH / "xmls" / "g1_mjx_feetonly.xml")

    arena = mujoco.MjSpec()
    arena.modelname = "motionforge_two_g1_flat"
    arena.option.timestep = timestep
    arena.option.iterations = 3
    arena.option.ls_iterations = 5

    arena.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.01],
    )

    prefixes = ("agent0/", "agent1/")
    for index, prefix in enumerate(prefixes):
        attachment = arena.worldbody.add_frame(name=f"agent{index}_attachment")
        robot = mujoco.MjSpec.from_file(robot_path, assets=assets)
        arena.attach(robot, prefix=prefix, frame=attachment)

    model = arena.compile()

    layouts: list[G1ModelLayout] = []
    for prefix in prefixes:
        root_joint_id = model.joint(f"{prefix}floating_base_joint").id
        qpos_start = int(model.jnt_qposadr[root_joint_id])
        qvel_start = int(model.jnt_dofadr[root_joint_id])
        actuator_ids = tuple(
            actuator_id
            for actuator_id in range(model.nu)
            if model.actuator(actuator_id).name.startswith(prefix)
        )

        if len(actuator_ids) != 29:
            raise RuntimeError(
                f"{prefix} has {len(actuator_ids)} actuators; expected 29"
            )

        layouts.append(
            G1ModelLayout(
                prefix=prefix,
                qpos_slice=slice(qpos_start, qpos_start + 36),
                qvel_slice=slice(qvel_start, qvel_start + 35),
                actuator_ids=actuator_ids,
            )
        )

    return TwoG1Model(model=model, agents=tuple(layouts))


def make_two_g1_data(
    model_bundle: TwoG1Model,
    separation: float = 2.0,
) -> mujoco.MjData:
    """Initialize both robots in the accepted knees-bent pose."""
    if separation <= 0.0:
        raise ValueError("separation must be positive")

    assets = base.get_assets()
    scene_path = Path(g1_constants.FEET_ONLY_FLAT_TERRAIN_XML)
    reference_model = mujoco.MjModel.from_xml_string(
        scene_path.read_text(encoding="utf-8"),
        assets=assets,
    )
    reference_keyframe = reference_model.key("knees_bent")

    data = mujoco.MjData(model_bundle.model)
    spawn_y = (-separation / 2.0, separation / 2.0)

    for agent, y_position in zip(
        model_bundle.agents,
        spawn_y,
        strict=True,
    ):
        pose = reference_keyframe.qpos.copy()
        pose[1] = y_position
        data.qpos[agent.qpos_slice] = pose

        for actuator_id, target in zip(
            agent.actuator_ids,
            reference_keyframe.ctrl,
            strict=True,
        ):
            data.ctrl[actuator_id] = target

    mujoco.mj_forward(model_bundle.model, data)
    return data
