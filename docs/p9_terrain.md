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

## Frozen-policy slope sweep

The population evaluator now accepts `pursuer_generation`,
`evader_generation`, and `slope_degrees` Hydra overrides. Its defaults still
evaluate every registered pairing on flat ground, while the filters permit a
targeted terrain test without recomputing the historical matrix.

The first generation-two sweep used identical seeds 3000--3003 at -10, -5,
0, 5, and 10 degrees. The five machine-readable results are stored as
`logs/p9/tag_generation2_slope_{m10,m5,flat,p5,p10}_a.json`. Every rollout
remained finite, loaded the requested immutable snapshots, and reached a
defined terminal outcome. Outcomes were:

| slope | tags | timeouts | pursuer falls | evader falls |
| ---: | ---: | ---: | ---: | ---: |
| -10 degrees | 3 | 0 | 1 | 0 |
| -5 degrees | 4 | 0 | 0 | 0 |
| 0 degrees | 1 | 3 | 0 | 0 |
| 5 degrees | 3 | 0 | 0 | 1 |
| 10 degrees | 0 | 0 | 2 | 2 |

These results do not establish slope robustness. Even five degrees changes
the flat-trained matchup qualitatively, and ten degrees causes physical
failure in every rollout. The next P9 experiment must expose slope selection
to training and begin with a symmetric low-angle curriculum. Heightfields,
bumps, gaps, and steps remain deferred.
