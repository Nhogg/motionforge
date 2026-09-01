from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import tyro
from dm_control.locomotion import soccer


@dataclass(frozen=True)
class Config:
    """Test intentional contact between two humanoids."""

    seed: int = 0
    duration: float = 0.75
    initial_separation: float = 0.8
    impact_speed: float = 1.5
    output: Path | None = None


def array_digest(array: np.ndarray) -> str:
    canonical = np.asarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def body_namespace(
    model: mujoco.MjModel,
    geom_id: int,
) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    if body_name is None:
        return ""

    return body_name.split("/", maxsplit=1)[0]


def is_player_contact(
    model: mujoco.MjModel,
    geom1: int,
    geom2: int,
) -> bool:
    namespace1 = body_namespace(model, geom1)
    namespace2 = body_namespace(model, geom2)

    return {namespace1, namespace2} == {
        "home0",
        "away0",
    }


def run_test(config: Config) -> dict[str, object]:
    if config.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")

    if config.initial_separation <= 0.0:
        raise ValueError("--initial-separation must be greater than zero")

    if config.impact_speed <= 0.0:
        raise ValueError("--impact-speed must be greater than zero")

    environment = soccer.load(
        team_size=1,
        time_limit=max(config.duration, 1.0),
        random_state=np.random.RandomState(config.seed),
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=soccer.WalkerType.HUMANOID,
    )
    environment.reset()

    model = environment.physics.model.ptr
    data = environment.physics.data.ptr

    home_joint = model.joint("home0/")
    away_joint = model.joint("away0/")

    home_qpos_address = int(home_joint.qposadr[0])
    away_qpos_address = int(away_joint.qposadr[0])
    home_dof_address = int(home_joint.dofadr[0])
    away_dof_address = int(away_joint.dofadr[0])

    # Preserve each humanoid's initialized pose and height, but place
    # them symmetrically on the x-axis.
    data.qpos[home_qpos_address] = -config.initial_separation / 2.0
    data.qpos[home_qpos_address + 1] = 0.0

    data.qpos[away_qpos_address] = config.initial_separation / 2.0
    data.qpos[away_qpos_address + 1] = 0.0

    # Remove reset velocities and send the roots toward one another.
    data.qvel[:] = 0.0
    data.qvel[home_dof_address] = config.impact_speed
    data.qvel[away_dof_address] = -config.impact_speed

    # Zero actuator commands: this is a contact test, not a policy test.
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)

    initial_qpos = data.qpos.copy()
    initial_time = float(data.time)

    physics_timestep = float(model.opt.timestep)
    simulation_steps = int(round(config.duration / physics_timestep))

    player_contact_steps = 0
    player_contact_count = 0
    maximum_simultaneous_player_contacts = 0
    maximum_contact_force = 0.0
    maximum_absolute_acceleration = 0.0
    minimum_player_contact_distance = math.inf
    state_is_finite = True

    first_player_contact_time: float | None = None

    for _ in range(simulation_steps):
        mujoco.mj_step(model, data)

        step_player_contacts = 0

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]

            if not is_player_contact(
                model,
                contact.geom1,
                contact.geom2,
            ):
                continue

            step_player_contacts += 1
            player_contact_count += 1

            minimum_player_contact_distance = min(
                minimum_player_contact_distance,
                float(contact.dist),
            )

            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                model,
                data,
                contact_index,
                contact_force,
            )

            maximum_contact_force = max(
                maximum_contact_force,
                float(np.linalg.norm(contact_force[:3])),
            )

        if step_player_contacts > 0:
            player_contact_steps += 1

            if first_player_contact_time is None:
                first_player_contact_time = float(data.time)

        maximum_simultaneous_player_contacts = max(
            maximum_simultaneous_player_contacts,
            step_player_contacts,
        )

        maximum_absolute_acceleration = max(
            maximum_absolute_acceleration,
            float(np.max(np.abs(data.qacc))),
        )

        state_is_finite &= bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        )

    elapsed_time = float(data.time) - initial_time

    checks = {
        "contact_force_finite": math.isfinite(maximum_contact_force),
        "player_contact_occurred": player_contact_count > 0,
        "player_contact_spanned_steps": player_contact_steps > 0,
        "simulation_time": math.isclose(
            elapsed_time,
            config.duration,
            rel_tol=0.0,
            abs_tol=physics_timestep,
        ),
        "state_finite": state_is_finite,
    }

    result = {
        "checks": checks,
        "dm_control_version": version("dm-control"),
        "duration": config.duration,
        "elapsed_time": elapsed_time,
        "experiment": "two_humanoid_contact_stability",
        "final_qpos_sha256": array_digest(data.qpos),
        "final_qvel_sha256": array_digest(data.qvel),
        "first_player_contact_time": first_player_contact_time,
        "impact_speed_per_humanoid": config.impact_speed,
        "initial_qpos_sha256": array_digest(initial_qpos),
        "initial_separation": config.initial_separation,
        "maximum_absolute_acceleration": (maximum_absolute_acceleration),
        "maximum_contact_force": maximum_contact_force,
        "maximum_simultaneous_player_contacts": (maximum_simultaneous_player_contacts),
        "minimum_player_contact_distance": (
            None
            if minimum_player_contact_distance == math.inf
            else minimum_player_contact_distance
        ),
        "mujoco_version": mujoco.__version__,
        "passed": all(checks.values()),
        "physics_timestep": physics_timestep,
        "player_contact_count": player_contact_count,
        "player_contact_steps": player_contact_steps,
        "python_version": platform.python_version(),
        "seed": config.seed,
        "simulation_steps": simulation_steps,
    }

    environment.close()
    return result


def main(config: Config) -> None:
    result = run_test(config)
    serialized = json.dumps(result, sort_keys=True)

    print(serialized)

    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(
            serialized + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
