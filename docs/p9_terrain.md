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

## Symmetric low-angle self-play curriculum

Slope orientation is part of the compiled MuJoCo model and cannot be changed
independently for each reset in one vectorized training job. The curriculum is
therefore scheduled at the self-play-round level. Both TAG training launchers
accept `slope_degrees`, and the alternating round runner forwards and records
that value in each role's command and round manifest. Flat terrain remains the
zero-degree default.

The first symmetric curriculum resumed generation two for two successive
rounds with the same requested budget of 131,072 steps per role:

- `logs/p9/rounds/round_0003_slope_p2p5_a` trained at +2.5 degrees and
  produced `pursuer_g0003_96b3a8320265` and
  `evader_g0003_576712741c6e`.
- `logs/p9/rounds/round_0004_slope_m2p5_a` resumed generation three at -2.5
  degrees and produced `pursuer_g0004_0b6630f65965` and
  `evader_g0004_46a096665267`.

Every role update completed 143,360 actual environment steps, recorded finite
metrics, verified its frozen opponent identity, and published a digest-checked
snapshot. A fresh four-seed retention evaluation on seeds 4000--4003 then
tested the generation-four pairing at both trained slopes and on flat ground:

| slope | tags | timeouts | boundary exits | falls | pursuer / evader wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -2.5 degrees | 2 | 2 | 0 | 0 | 50% / 50% |
| 0 degrees | 2 | 2 | 0 | 0 | 50% / 50% |
| 2.5 degrees | 1 | 2 | 1 evader | 0 | 50% / 50% |

The artifacts are
`logs/p9/tag_generation4_slope_{m2p5,flat,p2p5}_a.json`. This establishes
initial low-angle retention without loss of flat-ground balance. The sample is
still small, and the +2.5-degree boundary exit should remain visible in later
evaluations. The next curriculum increment may test symmetric 5-degree rounds;
heightfields remain premature until those policies avoid the failures seen in
the frozen generation-two 5-degree sweep.

## Five-degree curriculum failure

The next symmetric pair increased the fixed slopes to +5 and -5 degrees:

- `logs/p9/rounds/round_0005_slope_p5_a` produced
  `pursuer_g0005_a6d0a7ee10a0` and `evader_g0005_8b515adb7d49`.
- `logs/p9/rounds/round_0006_slope_m5_a` produced
  `pursuer_g0006_f2af84d41440` and `evader_g0006_3eee5003ed52`.

All four role updates completed 143,360 environment steps with finite
training metrics and verified opponent identities. That infrastructure result
must not be confused with policy success. The generation-six training
evaluations already contained one pursuer fall and one evader fall at -5
degrees.

A held-out sweep on seeds 6000--6003 rejected generation six as a robust
endpoint:

| slope | tags | timeouts | pursuer failures | evader failures |
| ---: | ---: | ---: | ---: | ---: |
| -5 degrees | 0 | 0 | 1 fall | 3 falls |
| -2.5 degrees | 0 | 0 | 0 | 4 boundary exits |
| 0 degrees | 0 | 0 | 0 | 4 boundary exits |
| 2.5 degrees | 2 | 0 | 0 | 2 boundary exits |
| 5 degrees | 2 | 0 | 0 | 1 fall, 1 boundary exit |

The artifacts are
`logs/p9/tag_generation6_slope_{m5,m2p5,flat,p2p5,p5}_a.json`. No episode
timed out, flat-ground balance was lost, and every -5-degree episode ended in
a fall. This is evidence of curriculum-induced gameplay and physical
regression, not successful adaptation.

Fixed-slope rounds remain useful controlled experiments, but sequentially
training one slope per job permits last-stage forgetting. Generation six is
retained as negative evidence and must not replace generation four as the
accepted slope policy. Before increasing difficulty or adding heightfields,
training must mix flat and signed slopes within one optimization run, or use
an equivalent replay schedule that repeatedly revisits all accepted terrain
stages.

## Per-reset mixed-slope terrain

The shared plane now belongs to a kinematic MuJoCo mocap body. Its orientation
is therefore part of each environment's MJX data rather than immutable model
data. `slope_curriculum_degrees=5` samples one of -5, 0, or +5 degrees from
the explicit JAX reset key, updates the floor quaternion, and adjusts both
agents' root heights to preserve terrain-relative clearance. Fixed
`slope_degrees` remains available for controlled evaluation, and the fixed and
curriculum modes are mutually exclusive.

Both role trainers and the self-play round runner expose and record the new
curriculum setting. The runner also accepts explicit restore-generation
metadata, allowing a repair branch to resume an older accepted checkpoint
without falsely labeling it as the immediately preceding generation.

`scripts/tests/test_tag_slope_curriculum.py` records its accepted audit at
`logs/p9/tag_slope_curriculum_a.json`. Across seeds 0--11 it observed every
terrain choice, reproduced the first reset exactly, retained equal 0.755-meter
root clearances, and produced finite GPU states. Flat, fixed-slope,
deterministic-reset, and self-play-plan regressions also passed.

### First mixed-slope repair round

`logs/p9/rounds/round_0007_mixed_slope5_repair_a` resumed accepted generation
four, not failed generation six, and trained both roles for 143,360 actual
steps with the three-way mixed slope sampler. It produced
`pursuer_g0007_b2f55d147420` and `evader_g0007_d1c4ef996307`. Training metrics
were finite and checkpoint/opponent identities passed, but the evader
evaluation still contained falls and no timeouts.

The held-out fixed-angle evaluation on seeds 7000--7003 confirmed that this
single short repair round was insufficient:

| slope | tags | timeouts | pursuer failures | evader failures |
| ---: | ---: | ---: | ---: | ---: |
| -5 degrees | 0 | 0 | 1 fall | 3 falls |
| -2.5 degrees | 0 | 1 | 0 | 3 boundary exits |
| 0 degrees | 0 | 0 | 0 | 4 boundary exits |
| 2.5 degrees | 0 | 0 | 0 | 4 boundary exits |
| 5 degrees | 0 | 0 | 2 falls | 1 fall, 1 boundary exit |

These results are stored in
`logs/p9/tag_generation7_mixed_slope_{m5,m2p5,flat,p2p5,p5}_a.json`.
Per-reset terrain mixing fixes the exposure architecture, but generation seven
is not an accepted policy. Generation four remains the accepted checkpoint.
The next repair should use a larger optimization budget and current opponents
instead of adding terrain complexity.
