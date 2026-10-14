# TAG strategy shaping and time-limit truncation

**Shaping superseded:** the clipped term below is historical. Array 4296 was
canceled with user approval and replaced by the
[distance-potential v2 protocol](2026-10-14-tag-distance-potential.md).
The truncation and terminal-cause refactor remains in effect.

## Scope and protocol

`strategy_reward_version = tag_progress_truncation_v1` identifies new training
runs. Locomotion reward, model architecture, role encoding, stop-gradient,
execution rates, optimizer schedules, Elo formula and league composition are
unchanged. The current strategy retains its shared encoder and existing separate
role heads; this patch does not replace the architecture with a single head.
Running training processes retain the old imported code. Evaluation processes
launched later use the updated protocol and record
`evaluation_protocol = tag_termination_truncation_v1`; do not silently pool old
evader win-rate fields with new censored-outcome fields.

## Reward

The valid TAG component remains exactly pursuer +1 and evader -1. No tag reward
is paid for an opponent fall/OOB. Own persistent fall and own OOB each cost 1.
Simultaneous own failures retain both penalties; no opposing-agent tag bonus is
created. Timeout contributes zero terminal reward for both roles.

At each strategy interval (normally five 50 Hz control transitions), progress is
computed from the interval's initial and last valid planar distances:

```text
pursuer_progress = .02 * clip((d_previous - d_current) / .10, -1, 1)
evader_progress = -pursuer_progress
```

Clipping happens once at strategy cadence, not five times at motor cadence.
Partial intervals ending in termination/truncation use their actual last valid
state. Components are stored at that last valid transition, then summed into
the strategy batch. The total and component streams aggregate identically.
There is no proximity bonus.

Command smoothness is -.005 times squared change in the sampled strategy
command, comparing like-for-like commands before boundary projection. It is
evaluated once per strategy interval, using zero as the initial previous command.
Boundary reward has the same function and configuration for both roles: square
arena clearance, onset 1m from the edge, weight .005, cap .02. The penalty is
`-min(cap, weight * clip(1-clearance/onset, 0, 1)^2)`. Thus the default maximum
actually attained is .005; the independent cap protects configurations with
larger weights. The existing deterministic command projection is unchanged.

Override configuration uses the requested nested keys:

```text
tag_reward.progress.enabled=true
tag_reward.progress.weight=0.02
tag_reward.progress.distance_scale_m=0.10
tag_reward.command_smoothness_weight=0.005
tag_reward.own_fall_penalty=1.0
tag_reward.own_oob_penalty=1.0
```

The per-step cap does NOT mathematically guarantee cumulative shaping <1: a
10s/10Hz rollout has 100 strategy intervals and a worst-case absolute progress
budget of 2.0. The specified formula is preserved, not silently supplemented
with an episode budget. W&B records signed sums, absolute sums, per-batch
maximum absolute sums and `progress_at_least_tag_magnitude_rate`. Review these
before lengthening training. Smoothness and boundary sums are also logged.

## Ending semantics and bootstrap

The environment exposes `terminated`, `truncated`, and a unique primary cause.
Legacy `done` remains only as `terminated OR truncated` for stopping/resetting;
neither trainer uses it to decide bootstrap. Cause priority is:

```text
NONFINITE > PURSUER_FALL > EVADER_FALL > PURSUER_OOB > EVADER_OOB > TAG > TIME_LIMIT
```

Failure/contact ties are not valid TAGs. A true terminal event at the exact time
limit wins over truncation. Secondary failure flags are retained for both agents.
Time-limit truncation covers both configured episode cutoffs and a shorter
collector horizon. The collector retains the actual last valid observation;
padding does not advance the returned state or replace its bootstrap value.
Every collected episode has one final cause; the training logger asserts
exclusive flags and one ending, and saves per-environment cause/flag arrays.

True terminal transitions have zero bootstrap. Truncations use next-state value
for BOTH strategy and locomotion GAE, but stop the advantage trace across a reset.
The GAE helper accepts explicit per-transition `next_values` for concatenated
episodes; a truncated transition followed by a reset must bootstrap its own
final observation, not the reset observation. TAG collection currently has one
episode per environment/segment, so its final-state bootstrap suffices.

Nonfinite detection inspects simulator qpos/qvel and root/planar finite flags.
It terminates and retains the preceding finite physical observation for safe
diagnostics; no nonfinite trajectory receives a tag or progress bonus. Fatal
CUDA/kernel errors cannot be recovered as ordinary JAX environment transitions.

## Diagnostics and outcome reporting

W&B logs `reward/{pursuer,evader}/{progress,tag,command_smoothness,boundary,fall,oob,total}`,
with episode signed/absolute sums and maxima. `metrics.jsonl` also records actual
per-environment component episode sums and final causes/flags, indexed by
environment ID, plus the rollout episode ID. Cause codes are in the manifest.

Pursuer diagnostics: initial/final/minimum distance, distance closed, time below
1m and .5m, tag rate and time-to-tag conditional on successful tags (zero if
there are none; inspect the accompanying success count).
Evader diagnostics: initial/final/maximum distance, separation gain, time near
boundary, and truncation rate. Both roles log rates for every ending cause.
Persistent fall rates are separate from transient near-fall/tilt observations.

Evaluation reports tag rate and physical survival separately. Its
`role_success` is null for time limits/physical failures, rather than declaring
an evader win or pursuer loss. Aggregate role success is conditional on valid
TAG games; it is not an unconditional win probability. Use tag/truncation rates
and survival for performance comparisons.

Elo previously inferred outcomes from reward totals. That is invalid under
shaping. The formula is unchanged, but censored trajectories do not generate
rating updates; the current update path uses valid TAG outcomes only and skips
an update if either role lacks them. A non-mutating rating snapshot supplies
plot keys for skipped batches. This censored rating is NOT a calibrated broad
win-probability measure; Elo matchmaking remains a separate future experiment.

## Verification and diagnostic experiment

CPU regression tests cover exact TAG payoff, zero timeout payoff, symmetric
progress signs/clipping, own failures, symmetric regularizers, ending priority,
truncation bootstrap, terminal bootstrap masking, padded final observations,
and trace isolation across resets. Synthetic JAX collector fixtures exercise
strategy-cadence clipping and the new episode diagnostics. Existing optimizer,
metrics, and Elo formula regressions are retained.

GPU suite: `scripts/slurm/hierarchical_tag_reward_smoke_g4.sbatch`. It exercises
actual Warp padded truncations, both trainer bootstrap paths, balanced batches,
frozen locomotion parameters/optimizer state, videos, and a one-iteration CLI run.
Initial GPU attempts exposed checkpoint-path/minibatch setup mistakes in the
new harness, then a packed-contact-buffer issue in cutoff-state selection.
The collector now uses Warp-native Data.where for physical state instead of
generic tree selection. The latest rerun is job 4295.
Job 4295 completed successfully: actual scalar and balanced next-state
bootstraps, padded cutoff observation retention, frozen locomotion parameters
and optimizer, existing rollout/PPO/batched tests, corrected video/trajectory
callback, and one-iteration CLI training with finite metrics all passed.
CPU shaping/ending/GAE/cadence tests, unchanged Elo/optimizer/PPO/role-head
regressions, Ruff lint and shell syntax checks also passed.

Diagnostic launcher: `scripts/slurm/hierarchical_tag_reward_diagnostic_g4.sbatch`.
Eight one-GPU tasks on g4: four fresh strategy seeds, each paired with progress
enabled/disabled. Both arms use correct truncation plus identical failure,
smoothness and boundary terms. Pretrained P2 locomotion is frozen. Historical
mixture stays 75% historical / 8.33% each current, stationary, scripted. No Elo
matchmaking or frequency changes. Train 100 iterations / 204.8M steps and
evaluate initialization/final checkpoints against self-play and historical
005/020, both roles/action modes, at fixed 1.5m over 10s.

Submitted as Slurm array **4296** after the smoke suite succeeded. Do not
resubmit the example below unless intentionally starting another experiment.

The suite has succeeded; submit without an expired dependency:

```bash
cd /home/nhogg/motionforge
mkdir -p logs/slurm
ssh head0 'cd /home/nhogg/motionforge && sbatch scripts/slurm/hierarchical_tag_reward_diagnostic_g4.sbatch'
```

Compare paired tag rates, pursuit-distance traces, physical failure
rates and reward budgets across seeds. These arms isolate the progress term
within the new ending protocol; comparing against older experiments changes
multiple reward/ending semantics and cannot isolate each component's effect.
