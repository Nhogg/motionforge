from __future__ import annotations

import json
import math
import platform
import random
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import mujoco
import tyro
from mujoco import mjx

MJCF = """
<mujoco model="falling_sphere">
<option timestep="0.002" gravity="0 0 -9.81"/>

<worldbody>
  <geom
    name="floor"
    type="plane"
    size="2 2 0.1"
  />

  <body name="ball" pos="0 0 1">
    <freejoint/>
    <geom
      name="ball_geom"
      type="sphere"
      size="0.05"
      mass="1"
    />
  </body>
</worldbody>
</mujoco>
"""


@dataclass(frozen=True)
class Config:
    """Configuration for the MJX smoke test."""

    seed: int = 0
    steps: int = 1000
    output: Path | None = None


def run_smoke_test(config: Config) -> dict[str, Any]:
    if config.steps <= 0:
        raise ValueError("--steps must be greater than zero")

    random.seed(config.seed)

    cpu_model = mujoco.MjModel.from_xml_string(MJCF)
    mjx_model = mjx.put_model(cpu_model)
    initial_data = mjx.make_data(mjx_model)

    initial_height = float(initial_data.qpos[2])
    initial_contacts = initial_data._impl.ncon

    def step_once(
        data: mjx.Data,
        _: None,
    ) -> tuple[mjx.Data, jax.Array]:
        next_data = mjx.step(mjx_model, data)
        return next_data, next_data._impl.ncon

    def rollout(
        data: mjx.Data,
    ) -> tuple[mjx.Data, jax.Array]:
        return jax.lax.scan(
            step_once,
            data,
            xs=None,
            length=config.steps,
        )

    final_data, contact_counts = jax.jit(rollout)(initial_data)

    # Ensure execution is finished before extracting host values.
    final_data.qpos.block_until_ready()

    final_height = float(final_data.qpos[2])
    timestep = float(cpu_model.opt.timestep)
    simulation_time = float(final_data.time)
    expected_simulation_time = config.steps * timestep
    simulation_time_tolerance = timestep * 0.1
    maximum_contacts = int(jax.numpy.maximum(initial_contacts, contact_counts.max()))

    numeric_values = (
        initial_height,
        final_height,
        timestep,
        simulation_time,
        expected_simulation_time,
    )

    checks = {
        "finite_values": all(math.isfinite(value) for value in numeric_values),
        "initial_height": math.isclose(
            initial_height,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "resting_height": 0.049 <= final_height <= 0.051,
        "contact_occurred": maximum_contacts >= 1,
        "simulation_time": math.isclose(
            simulation_time,
            expected_simulation_time,
            rel_tol=0.0,
            abs_tol=simulation_time_tolerance,
        ),
    }

    return {
        "backend": jax.default_backend(),
        "checks": checks,
        "devices": [str(device) for device in jax.devices()],
        "experiment": "mjx_smoke_test",
        "expected_simulation_time": expected_simulation_time,
        "final_height": final_height,
        "initial_height": initial_height,
        "jax_version": jax.__version__,
        "maximum_contacts": maximum_contacts,
        "mujoco_mjx_version": version("mujoco-mjx"),
        "mujoco_version": mujoco.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "seed": config.seed,
        "simulation_time": simulation_time,
        "simulation_time_error": abs(simulation_time - expected_simulation_time),
        "simulation_time_tolerance": simulation_time_tolerance,
        "steps": config.steps,
        "timestep": timestep,
    }


def main(config: Config) -> None:
    result = run_smoke_test(config)
    serialized_result = json.dumps(result, sort_keys=True)

    print(serialized_result)

    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(serialized_result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(Config))
