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

### Longer mixed-slope repair round

`logs/p9/rounds/round_0008_mixed_slope5_repair_1m` repeated the repair from
accepted generation four with `current_opponent_probability=1.0` and
1,048,576 requested steps per role. Each role completed 1,064,960 actual
steps. The round passed its infrastructure checks and published
`pursuer_g0008_db6ee1b3ba8e` and `evader_g0008_8c97d32756ad`. The pursuer's
final training evaluation tagged all 32 opponents without falling or leaving
the arena. The evader's final evaluation, however, produced only one tag in
32 episodes and no timeouts; 12 episodes ended in an evader fall and 17 ended
at the boundary. Avoiding a tag by terminating the episode is therefore still
the dominant evader behavior.

Held-out fixed-angle evaluation used seeds 8000--8015 at every slope:

| slope | tags | timeouts | pursuer failures | evader failures |
| ---: | ---: | ---: | ---: | ---: |
| -5 degrees | 0 | 0 | 1 fall, 5 boundary exits | 8 falls, 2 boundary exits |
| -2.5 degrees | 3 | 0 | 0 | 13 boundary exits |
| 0 degrees | 0 | 0 | 0 | 16 boundary exits |
| 2.5 degrees | 0 | 0 | 0 | 16 boundary exits |
| 5 degrees | 3 | 0 | 3 falls, 1 boundary exit | 6 falls, 3 boundary exits |

The evaluation artifacts are
`logs/p9/tag_generation8_mixed_slope_{m5,m2p5,flat,p2p5,p5}_a.json`.
Increasing the optimization budget alone did not repair terrain robustness or
produce valid evasion. Generation eight is rejected, and generation four
remains the accepted checkpoint. Before further terrain training, the evader
objective or termination handling must be changed so falls and boundary exits
cannot serve as successful tag avoidance.

### Evader terminal-outcome ordering

The original evader reward assigned a penalty of 10 to being tagged, falling,
and leaving the arena. A failure could therefore tie a tag before considering
shaping terms, while moving toward the boundary could also collect positive
separation reward. This made intentional early termination a competitive
strategy.

The default fall and out-of-bounds penalties are now 25, while the tag penalty
and timeout reward remain 10. The evader training launcher exposes all four
values as Hydra fields, and the self-play round forwards and records them in
its manifest. Both launchers reject configurations in which either failure
penalty does not strictly exceed the tag penalty.

`logs/p9/tag_evader_failure_ordering_a.json` verifies on GPU that isolated
fall and boundary-exit returns are strictly lower than an isolated tag return.
`logs/p9/tag_self_play_failure_penalties_a.json` verifies that the self-play
runner constructs the intended evader training command. These checks validate
the reward contract only; a new held-out experiment is still required before
accepting a policy trained with it.

### Isolated evader reward repair

`logs/p9/training/evader_e4_reward_repair_1m_seed10` restored accepted E4,
froze accepted P4, and trained only the evader for 1,064,960 actual steps on
the per-reset three-slope mixture. The training revision was clean and the
launcher recorded the 10-point tag penalty, 25-point fall and boundary-exit
penalties, and 10-point timeout reward. Its final 32-episode evaluation
contained 16 tags, 2 evader falls, 14 evader boundary exits, and no timeouts.
The stronger terminal penalties reduced falling but did not by themselves
produce successful evasion or eliminate boundary exploitation.

The standalone evader evaluator now exposes `slope_degrees`, allowing this
unregistered repair checkpoint to be audited at fixed angles without adding a
rejected candidate to the self-play population. Fixed-angle held-out results
remain required before the repair is accepted or rejected conclusively.

The fixed-angle audit used seeds 10000--10015 against accepted P4:

| slope | tags | timeouts | pursuer failures | evader failures |
| ---: | ---: | ---: | ---: | ---: |
| -5 degrees | 2 | 2 | 6 falls | 3 falls, 3 boundary exits |
| -2.5 degrees | 2 | 2 | 2 boundary exits | 10 boundary exits |
| 0 degrees | 0 | 10 | 0 | 6 boundary exits |
| 2.5 degrees | 0 | 2 | 0 | 14 boundary exits |
| 5 degrees | 1 | 0 | 2 falls | 7 falls, 6 boundary exits |

The artifacts are
`logs/p9/tag_evader_reward_repair_{m5,m2p5,flat,p2p5,p5}_heldout16_a.json`.
All 80 rollouts were finite and completed. The repeated first-seed outcome
check passed at four angles but failed on flat terrain, so the flat artifact
also records an outcome-level repeatability warning near a terminal boundary.

The reward repair produced genuine flat-ground evasion and sharply reduced
falls near zero slope, but it did not generalize across terrain and remains
dominated by boundary exits. The checkpoint is rejected and remains outside
the self-play population. The next repair should add a denser boundary-control
signal or constrain commands near the arena edge rather than merely increasing
the terminal penalty again.

### Dense outward-boundary velocity penalty

The evader reward now adds a separately logged dense term inside the existing
one-meter boundary margin. For each planar axis it multiplies normalized edge
intrusion by the positive component of velocity pointing away from the arena
center, sums the two axes, and applies a default scale of 5. Motion in the safe
interior, tangential motion, and inward motion receive no penalty from this
term. The existing position-based boundary penalty and terminal failure
penalties remain unchanged.

The scale is exposed as the Hydra field
`boundary_outward_velocity_penalty_scale` in the evader trainer and as
`evader_boundary_outward_velocity_penalty_scale` in the self-play round. It is
therefore recorded in training manifests and forwarded reproducibly during
self-play. `logs/p9/tag_evader_boundary_velocity_reward_a.json` verifies the
directional reward semantics and integrated GPU step, while
`logs/p9/tag_self_play_boundary_velocity_a.json` verifies command forwarding.
No policy trained with this term has yet been accepted.

### Isolated dense-boundary repair experiment

`logs/p9/training/evader_e4_boundary_repair_1m_seed11` restored accepted E4,
froze accepted P4, and trained only the evader for 1,064,960 actual steps on
the per-reset {-5, 0, +5}-degree slope mixture. It retained the ordered
terminal rewards from the preceding repair and enabled the dense outward
boundary-velocity penalty at scale 5. The final 32-episode training evaluation
was finite, but every episode ended in a tag after 43 steps. The mean dense
boundary-velocity reward contribution was -1.334 per episode, confirming that
the signal was active; the policy nevertheless collapsed against the training
evaluation batch.

The fixed-angle audit used held-out seeds 11000--11015 against accepted P4:

| slope | tags | timeouts | pursuer failures | evader failures |
| ---: | ---: | ---: | ---: | ---: |
| -5 degrees | 0 | 2 | 5 falls | 4 falls, 5 boundary exits |
| -2.5 degrees | 0 | 1 | 2 boundary exits | 13 boundary exits |
| 0 degrees | 0 | 0 | 0 | 16 boundary exits |
| 2.5 degrees | 2 | 1 | 0 | 13 boundary exits |
| 5 degrees | 3 | 1 | 3 falls | 2 falls, 7 boundary exits |

The artifacts are
`logs/p9/tag_evader_boundary_repair_{m5,m2p5,flat,p2p5,p5}_heldout16_a.json`.
All 80 rollouts were finite and completed, and the repeated first-seed outcome
check passed at all five angles. Only 5 of 80 episodes timed out successfully;
54 ended in an evader boundary exit and 6 in an evader fall. The term changes
the optimization signal but does not prevent the learned policy from driving
outward, and its scale-5 result is worse than the preceding reward-only repair
on flat terrain.

This checkpoint is rejected and remains outside the self-play population.
Generation four remains the accepted terrain policy. Further attempts should
not merely increase the dense penalty or extend this failed checkpoint. The
next repair should make boundary safety part of command selection, for example
by projecting unsafe outward planar commands near an edge, and must keep that
mechanism modular and independently tested before another training run.
