from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import mediapy
import mujoco
import numpy as np
import tyro


@dataclass(frozen=True)
class Config:
    seed: int = 0
    steps: int = 2000
    timestep: float = 0.002
    output: Path = Path("logs/p1/heightfield_contact_a.json")
    view: bool = False
    video_output: Path = Path("logs/p1/heightfield_contact.mp4")
    capture_fps: int = 60
    width: int = 960
    height: int = 720


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def make_terrain() -> np.ndarray:
    coordinates = np.linspace(-1.0, 1.0, 9)
    x, y = np.meshgrid(coordinates, coordinates)

    elevation = 0.5 + 0.3 * x + 0.12 * np.cos(np.pi * x) * np.cos(np.pi * y)
    elevation -= elevation.min()
    elevation /= elevation.max()
    return elevation


def make_model_xml(terrain: np.ndarray, timestep: float) -> str:
    elevation_text = " ".join(f"{value:.17g}" for value in terrain.reshape(-1))

    drop_bodies = "\n".join(
        f"""
      <body name="drop_body_{index}" pos="{x_position} 0 0.8">
        <joint
          name="drop_joint_{index}"
          type="slide"
          axis="0 0 1"
        />
        <geom
          name="drop_geom_{index}"
          type="sphere"
          size="0.08"
          mass="1"
          friction="1 0.005 0.0001"
        />
      </body>
      """
        for index, x_position in enumerate((-1.0, 0.0, 1.0))
    )

    return f"""
  <mujoco model="heightfield_contact">
    <option timestep="{timestep:.17g}" gravity="0 0 -9.81" />

    <size nconmax="100" />

    <asset>
      <hfield
        name="terrain"
        nrow="{terrain.shape[0]}"
        ncol="{terrain.shape[1]}"
        size="2 2 0.25 0.1"
        elevation="{elevation_text}"
      />
    </asset>

    <worldbody>
      <geom
        name="terrain_geom"
        type="hfield"
        hfield="terrain"
        friction="1 0.005 0.0001"
      />

      {drop_bodies}
    </worldbody>
  </mujoco>
  """


def run_experiment(config: Config) -> dict[str, object]:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.timestep <= 0.0:
        raise ValueError("timestep must be positive")

    terrain = make_terrain()
    model = mujoco.MjModel.from_xml_string(make_model_xml(terrain, config.timestep))
    data = mujoco.MjData(model)

    terrain_geom_id = model.geom("terrain_geom").id
    drop_geom_ids = {model.geom(f"drop_geom_{index}").id: index for index in range(3)}

    initial_heights = [
        float(data.body(f"drop_body_{index}").xpos[2]) for index in range(3)
    ]

    contacted_spheres: set[int] = set()
    contact_steps = 0
    maximum_simultaneous_contacts = 0
    maximum_contact_force = 0.0
    minimum_contact_distance = 0.0
    maximum_absolute_acceleration = 0.0
    finite_values = True

    contact_force = np.zeros(6, dtype=np.float64)

    renderer = None
    frames: list[np.ndarray] = []
    next_frame_time = 0.0
    if config.view:
        if config.capture_fps <= 0:
            raise ValueError("capture_fps must be positive")
        if config.width <= 0 or config.height <= 0:
            raise ValueError("width and height must be positive")

        model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)
        renderer = mujoco.Renderer(
            model,
            width=config.width,
            height=config.height,
        )

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.2)
        camera.distance = 4.5
        camera.azimuth = 135.0
        camera.elevation = -25.0

    try:
        for _ in range(config.steps):
            mujoco.mj_step(model, data)

            finite_values = finite_values and bool(
                np.isfinite(data.qpos).all()
                and np.isfinite(data.qvel).all()
                and np.isfinite(data.qacc).all()
            )

            maximum_absolute_acceleration = max(
                maximum_absolute_acceleration,
                float(np.max(np.abs(data.qacc))),
            )

            simultaneous_contacts = 0

            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                geom_pair = {int(contact.geom1), int(contact.geom2)}

                if terrain_geom_id not in geom_pair:
                    continue

                sphere_geom_id = next(
                    (geom_id for geom_id in drop_geom_ids if geom_id in geom_pair),
                    None,
                )
                if sphere_geom_id is None:
                    continue

                simultaneous_contacts += 1
                contacted_spheres.add(drop_geom_ids[sphere_geom_id])
                minimum_contact_distance = min(
                    minimum_contact_distance,
                    float(contact.dist),
                )

                mujoco.mj_contactForce(
                    model,
                    data,
                    contact_index,
                    contact_force,
                )
                force_magnitude = float(np.linalg.norm(contact_force[:3]))
                finite_values = finite_values and bool(
                    np.isfinite(contact_force).all()
                )
                maximum_contact_force = max(
                    maximum_contact_force,
                    force_magnitude,
                )

            if simultaneous_contacts > 0:
                contact_steps += 1

            maximum_simultaneous_contacts = max(
                maximum_simultaneous_contacts,
                simultaneous_contacts,
            )

            if renderer is not None and data.time >= next_frame_time:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
                next_frame_time += 1.0 / config.capture_fps
    finally:
        if renderer is not None:
            renderer.close()

    if config.view:
        config.video_output.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(
            config.video_output,
            frames,
            fps=config.capture_fps,
        )

    final_heights = [
        float(data.body(f"drop_body_{index}").xpos[2]) for index in range(3)
    ]

    final_vertical_speeds = [
        abs(float(data.qvel[int(model.joint(f"drop_joint_{index}").dofadr[0])]))
        for index in range(3)
    ]

    expected_simulation_time = config.steps * config.timestep
    simulation_time_error = abs(float(data.time) - expected_simulation_time)
    simulation_time_tolerance = max(
        1.0e-9,
        config.steps * np.finfo(np.float64).eps * 100.0,
    )

    checks = {
        "all_spheres_contacted": len(contacted_spheres) == 3,
        "contact_force_finite": np.isfinite(maximum_contact_force).item(),
        "contacts_spanned_steps": contact_steps >= 10,
        "different_resting_heights": (max(final_heights) - min(final_heights) > 0.05),
        "simulation_time": (simulation_time_error <= simulation_time_tolerance),
        "state_finite": finite_values,
        "stable_resting_state": max(final_vertical_speeds) < 0.02,
    }

    return {
        "checks": checks,
        "contact_steps": contact_steps,
        "contacted_spheres": sorted(contacted_spheres),
        "expected_simulation_time": expected_simulation_time,
        "experiment": "mujoco_heightfield_contact_stability",
        "final_heights": final_heights,
        "final_qpos_sha256": array_sha256(data.qpos),
        "final_qvel_sha256": array_sha256(data.qvel),
        "final_vertical_speeds": final_vertical_speeds,
        "initial_heights": initial_heights,
        "maximum_absolute_acceleration": maximum_absolute_acceleration,
        "maximum_contact_force": maximum_contact_force,
        "maximum_simultaneous_contacts": (maximum_simultaneous_contacts),
        "minimum_contact_distance": minimum_contact_distance,
        "mujoco_version": mujoco.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "seed": config.seed,
        "simulation_time": float(data.time),
        "simulation_time_error": simulation_time_error,
        "simulation_time_tolerance": simulation_time_tolerance,
        "steps": config.steps,
        "terrain_shape": list(terrain.shape),
        "terrain_sha256": array_sha256(terrain),
        "timestep": config.timestep,
        **(
            {
                "video_frames": len(frames),
                "video_output": str(config.video_output),
            }
            if config.view
            else {}
        ),
    }


def main(config: Config) -> None:
    result = run_experiment(config)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, sort_keys=True))

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Config))
