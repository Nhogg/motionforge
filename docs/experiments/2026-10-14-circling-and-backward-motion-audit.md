# Circling / backward-motion diagnosis and staged experiments

## Scope

User authorized both g3 and g4 for diagnosis and promising controlled experiments.
No existing jobs were cancelled. Both nodes were idle at submission. Keep valid
TAG payoff +1/-1, timeout truncation bootstrapping, and frozen accepted locomotion.
Changes to initialization and reward coefficients below are explicit experimental
arms, not changes to the established default game or retrospective metrics.

## First measured findings

GPU audit smoke **4322** passed. Synthetic CPU motion frame/sign/alignment tests
and all 12 reward/truncation tests passed. Full audit **4323** runs one GPU per
checkpoint (four seeds x shaping enabled/disabled) on g3.

At final checkpoint 100, deterministic self-play reset seed 3000:

- Potential-shaped seed 2 pursuer: backward command fraction 100%, actual
  backward-motion fraction 98.6%, mean commanded yaw -0.983 rad/s, actual yaw
  -0.542 rad/s, absolute heading turns .863, path efficiency .172. Actual
  motion toward the evader averages -0.136 m/s. It is requesting retreat and
  turning; this is not just an animation artifact.
- Potential-shaped seed 1 pursuer: mean commanded yaw .874 rad/s, actual .531,
  .850 turns in 10 seconds, path efficiency .163, actual radial approach
  approximately .001 m/s. This is quantitatively consistent with circling.
- Relative-position heading-frame reconstruction errors in the first eight
  learned trajectories are approximately .0016–.0046 m. This does not suggest
  a gross sign/90-degree/world-heading transform error in those examples.
  It is not proof that every observation/sensor convention is correct.
- Boundary projection is essentially inactive for the shaped seed-1 pursuer
  and both seed-2 pursuer examples, so it does not explain those particular
  circular/retreat trajectories.
- Zeroing yaw for **both** agents rescues one no-shaping seed-1 trajectory but
  does not generally rescue the others. It is a pair-level intervention, not
  an isolated pursuer causal test. Some zero-yaw rollouts end OOB.
- Scripted pursuit raises actual pursuer radial approach to roughly .61–.75
  m/s for several failed pairs, but still does not guarantee final contact
  against the moving learned evader. Motor tracking is imperfect; failure
  cannot be assigned wholly to either hierarchy level yet.
- Successful seed-0 pairs have both agents moving toward each other. A high
  self-play tag rate alone is not evidence of competent evasion or both roles
  improving. Keep role-separated fixed-opponent evaluation.
- Completed fixed-command probes across all eight checkpoints give straight
  forward paths (path efficiency .971) and tags from 3 m in approximately
  3.24–3.28 seconds; zero-yaw actual drift is near zero. Backward probes likewise
  follow predominantly straight paths (efficiency .958–.959), with backward
  motion fraction .986. Yaw-only command .5 rad/s yields approximately .447
  rad/s actual turning; standing motion RMSE is approximately .005 m/s.
  This is evidence that basic command signs and the motor adapter work in the
  two-agent simulation. It does not establish accurate combined-command tracking.
  Backward probes eventually terminate OOB despite command projection; the
  projection is not an invariant safety guarantee against drift/inertia.

Metrics use finite-difference world root translation and yaw; startup transients
are included. Command at stored sample t+1 drove physical interval t -> t+1.
Backward motion is velocity along body heading below -.05 m/s; it does not by
itself mean moving away from the opponent. Both quantities are reported.

## Motion audits: array 4323, g3, 8 GPUs

Completed all 88 probes and W&B uploads. For each old final checkpoint: two reset seeds with learned, zero-yaw (both
agents), and scripted-pursuer commands; one learned stochastic rollout; fixed
forward, backward, yaw-only, and standing commands for both robots at 3 m.
Each rollout saves world paths, root orientation, raw/executed commands, TAG
ending cause, tracking errors, radial speeds, turning and backward fractions.
Fixed commands and scripted pursuit are evaluation-only interventions.

Paths: `logs/hierarchical/motion_audit/job4323/`.
W&B group: `tag_motion_audit`; upload occurs when each task finishes all probes.
Diagnostic interventions must never be presented as learned-policy performance.

## Mechanism training: array 4324, g4, 8 GPUs

Two previously problematic training seeds (1,2), four arms, 100 iterations /
204.8M world steps, fresh strategy and frozen known-good locomotion:

| Arm | Difference from potential025 control |
|---|---|
| control | alpha .25, command smoothness .005, GAE lambda .95 |
| no_smoothness | command smoothness coefficient 0 |
| potential050 | alpha .50 |
| gae100 | GAE lambda 1.0 |

Same historical/current/curriculum mixture, optimizer learning rate, architecture,
reset distance and frequencies. First iteration was recorded with finite metrics
in all eight tasks. W&B group `tag_pursuit_mechanism`.

## Initialization follow-up: smoke 4339, array 4340, g3

GPU smoke 4339 passed its training completion checks; all eight tasks in 4340
were verified RUNNING on g3 after the audits finished. Array requires successful smoke 4339.
Two seeds (1,2), each with:

- command-head initialization scale .01;
- command-head initialization scale .10;
- original initialization, lower strategy learning rate 1e-4;
- scale .01 plus no command-smoothness penalty (explicit interaction arm).

Compare with standard-init controls in 4324. The isolated experiment launcher
scales only freshly initialized strategy command kernels/biases. Value heads,
encoders, motor weights, architecture and checkpoint restoration are unchanged.
The CPU check confirms those other parameters and the input tree are untouched.
Do not assume saturation was caused by initialization until initial-checkpoint
and final-command diagnostics support that conclusion.
W&B group `tag_initialization_diagnostic`. No training videos in this follow-up;
saved checkpoints permit subsequent trajectory/video evaluation.

## Evaluation and decision rules

Training sweeps evaluate iteration 0 and 100 against the same development
historical opponents 005/020 at 1.5 and 2 m, both action modes, 16 reset seeds,
both learner roles. Current-policy self-play is a separate changing-opponent
probe, not a fixed skill benchmark. Historical 005/020 are in the training
league; passing them is not held-out generalization. Before promoting a winner,
evaluate reserved independent opponents and broader starting configurations.

Compare role-separated tags/survival, pursuit distance closure/minimum distance,
command/yaw saturation, component returns, and ending causes. A useful arm must
improve more than one cherry-picked video or training seed. If forward/backward
fixed-command probes fail in the two-agent world, investigate the observation,
normalization, gait phase and actuator adapter against standalone P2 before
expanding strategy training. If fixed probes work but strategies request circles,
prioritize initialization, credit assignment, command regularization and steering.
No fabricated skill/difficulty metric is used to select an arm.

## CPU actor-response follow-up during training

`scripts/evaluation/audit_tag_strategy_feedback.py` verifies checkpoint SHA256
digests, runs only on CPU, and probes 32 synthetic target bearings for each role
with previous command zero versus 50 closed-loop previous-command feedback steps.
These are fixed observations, **not physical rollouts or task performance**.
The machine-readable output is
`logs/hierarchical/diagnostics/overnight/strategy_feedback.json`.

For old potential025 pursuers, initialization has no yaw commands above .9 rad/s
on these probes. At iteration 100, seed 2 requests above .9 in magnitude for all
32 bearings even without feedback. Seed 3 crosses that limit for all bearings
after feedback. Seed 2's feedback yaw has bearing standard deviation about .001
rad/s despite target bearing spanning a circle. Other trained policies also
have weak bearing sensitivity relative to their strong preferred yaw direction.

This narrows the hypothesis: extreme turning is learned, not simply present at
initialization, and previous-command input can amplify the learned bias. It is
not proof that the input should be removed. Smaller initialization is still a
valid ablation of the optimization trajectory; do not claim it fixes an already
saturated initial actor.

Code review also notes that the current updater accumulates minibatch gradients
and performs one optimizer step, normalizes advantages within contiguous chunks,
and averages chunk means without weighting by valid-sample count. Padding and
role/episode grouping can therefore influence the effective training objective.
Before changing this, reproduce the gradient/weighting effect with controlled
CPU fixtures and compare an isolated implementation against the active control.
Do not silently modify running experiments or assert this explains failures yet.

### First intermediate checkpoints (iteration 25)

CPU probes of newly saved 4324 checkpoints reproduce the old control's steering
pattern: seed-2 control feedback mean absolute yaw .919 rad/s, all 32 synthetic
bearings above .9 magnitude, bearing-response standard deviation only .010.
Doubling potential strength gives nearly the same .918 rad/s response; disabling
command smoothness gives .908 rad/s and 22/32 bearings above .9. Thus those two
changes have not prevented early steering saturation in this probe.

GAE lambda 1.0 produces lower mean absolute yaw at iteration 25 (.157 for seed 1,
.525 for seed 2) but the variable-opponent training tags are not consistently
better. This is not a success claim: less turning is insufficient if approach
and contact do not improve. Await matched physical evaluations and final
checkpoints before choosing an arm. Initialization sweep checkpoints had not
yet reached iteration 25 at this observation.

## Native CPU motor feasibility and migration audit

`scripts/evaluation/audit_tag_motor_commands_cpu.py` completed 11 fixed-command
probes, each with .5 seconds standing warmup and 2 seconds commanded motion.
No persistent falls/non-finite states occurred. Production observation ordering
and checkpoint SHA256 were verified. This is native MuJoCo, deterministic motor
feasibility, not Warp performance or a competitive TAG evaluation.

Forward/backward commands of +/- .8 m/s produced +.716/-.781 m/s. Pure yaw
1 rad/s produced .850 rad/s, whereas simultaneous translation and yaw produced
about .62–.70 rad/s with yaw RMSE .39–.46 rad/s. Mean joint-action saturation
fraction above .95 stayed at or below 1.38%. Combined tracking is imperfect
without gross saturation; this does not establish the cause of learned circling.

Migrated deterministic actions exactly match the original P2 controller at the
tested standing snapshot (maximum difference 0). Original pre-tanh action std
min/mean/max was .0342/.2375/.7548; migration initializes uniform .1353 std.
Thus deterministic means are preserved but the original state-dependent noise
distribution is not. Training samples motor actions even with frozen weights.
This is a confirmed distribution mismatch, not a proven cause of strategy failure;
the deterministic feasibility probes do not test either stochastic distribution.

Artifacts: `logs/hierarchical/diagnostics/overnight/motor_command_feasibility.{json,md}`.
An independent content audit verified 24 saved checkpoints retain iteration-0
motor parameters and optimizer state with zero motor updates. This verifies
freezing, not faithful reproduction of original P2 stochastic behavior.

## Halfway actor probes (iteration 50)

The CPU synthetic-bearing report now includes checkpoints 0/25/50/75/100.
At iteration 50, seed-2 control, no-smoothness, and alpha-.50 pursuers still
request mean absolute yaw .978/.968/.977 rad/s, exceeding .9 for all bearings.
Small-initialization arms avoid this saturation (mean absolute yaw .156–.470
across the two seeds and three small-init variants), but yaw response standard
deviation across bearings remains only .0002–.0077 rad/s. Their mean role-correct
radial command averaged over the bearing circle remains approximately zero.
GAE-1.0 pursuers have mean absolute yaw .326/.805 and modest positive mean
role-correct radial command .023/.017 m/s. These are actor-only probes, not
physical success evidence. Do not select a treatment from these figures alone.

## Paired native CPU motor-noise comparison

`audit_tag_motor_noise_cpu.py` completed 27/27 trials: stand, forward, and
forward-plus-yaw; three Gaussian innovation seeds; deterministic, migrated
Gaussian, and original P2 state-dependent Gaussian execution. Each pair has
identical deterministic .5-second warmup and replays identical standard-normal
innovations during the 2-second command. Original and migrated deterministic
actions match exactly on all 3375 evolving observation batches. All trials
stayed finite without detected falls. These are three Monte Carlo noise seeds
of one frozen motor policy, **not three independently trained controllers**.

Agent-0 yaw-rate tracking RMSE, averaged over noise seeds (rad/s):

| Command | Deterministic | Migrated noise | Original P2 noise |
|---|---:|---:|---:|
| Stand | .022 | .714 | .270 |
| Forward | .168 | .634 | .526 |
| Forward + yaw | .388 | .715 | .705 |

The replacement noise increases instantaneous yaw-rate jitter in this native
CPU test. It does not demonstrate sustained turning bias or strategic circling:
standing mean yaw rates stay close to zero relative to their RMSE, and the
learned circular trajectories already request strong nonzero yaw explicitly.
Original P2 noise is not uniformly better; combined-command yaw errors are
similar, and some translation errors increase. Do not restore noise blindly.

Next isolated mechanism to prepare: retain stochastic strategy exploration but
execute the frozen motor deterministically, with truthful separate action-mode
metadata. Keep existing default sampling and all current experiments unchanged.
Require a GPU smoke and matched physical evaluations before any competence claim.
Artifacts: `logs/hierarchical/diagnostics/overnight/motor_noise_feasibility.{json,md}`.
