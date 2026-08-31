from __future__ import annotations

import json
import math
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import tyro

# MuJoCo model descriptor
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
    """Config for standard MuJoCo test"""

    seed: int = 0

    steps: int = 1000

    output: Path | None = None


def run_smoke_test(config: Config) -> dict[str, Any]:
    if config.steps <= 0:
        raise ValueError("--steps must be greater than zero")

    random.seed(config.seed)

    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    initial_height = float(data.qpos[2])
    maximum_contacts = int(data.ncon)

    for _ in range(config.steps):
        mujoco.mj_step(model, data)
        maximum_contacts = max(maximum_contacts, int(data.ncon))

    final_height = float(data.qpos[2])
    timestep = float(model.opt.timestep)
    simulation_time = float(data.time)
    expected_simulation_time = config.steps * timestep

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
            abs_tol=1e-12,
        ),
        "resting_height": 0.049 <= final_height <= 0.051,
        "contact_occurred": maximum_contacts >= 1,
        "simulation_time": math.isclose(
            simulation_time,
            expected_simulation_time,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }

    return {
        "checks": checks,
        "experiment": "standard_mujoco_smoke_test",
        "expected_simulation_time": expected_simulation_time,
        "final_height": final_height,
        "initial_height": initial_height,
        "maximum_contacts": maximum_contacts,
        "mujoco_version": mujoco.__version__,
        "passed": all(checks.values()),
        "python_version": platform.python_version(),
        "seed": config.seed,
        "simulation_time": simulation_time,
        "steps": config.steps,
        "timestep": timestep,
    }


def main(config: Config) -> int:
    result = run_smoke_test(config)
    serialized_result = json.dumps(result, sort_keys=True)

    print(serialized_result)
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(serialized_result + "\n", encoding="utf-8")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    main(tyro.cli(Config))
