# P9 uneven and adversarial terrain

## Planar slope foundation

The first P9 change adds a single shared planar slope while preserving the
existing flat arena as the default. `build_two_g1_model` accepts a finite
`slope_degrees` value in the range -30 to 30 and rotates the floor about the
world y-axis. `TagEnvironmentConfig` carries the same setting into model
construction and deterministic reset sampling; no training or evaluation
script changes behavior unless it explicitly selects a nonzero slope.

Random tag resets require terrain-aware root heights. For a plane tilted by
angle `a` about the y-axis, the surface height is `-tan(a) * x`. The reset
therefore adds that height to each agent's accepted root height after sampling
its symmetric planar spawn. Both agents retain identical terrain-relative
clearance even when their randomized spawn axis has an x component.

`scripts/tests/test_tag_slope_terrain.py` provides the initial Hydra smoke
experiment. The accepted 5-degree run is stored at
`logs/p9/tag_slope_terrain_a.json`. It verified the compiled floor normal,
equal initial root clearances, finite GPU state, upright agents, and no early
termination across ten control steps. The existing flat-terrain and
deterministic-reset tests also pass unchanged. This validates the slope model
primitive only; learned-policy slope evaluation and a difficulty curriculum
must precede low-frequency heightfields.
