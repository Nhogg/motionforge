# Correcting flat skill/difficulty diagnostics

## What was wrong

The original trainer updated its role-biased game-Elo diagnostic only when both
learner roles had at least one TAG, then supplied pursuer-win rate 1 in both
matchups. Most non-TAG outcomes disappeared. Symmetric opposing rating changes
made the averaged skill/difficulty curves cancel to .5. That is not evidence of
a learning plateau. The old combined reward plot also added .5 to an average
of shaped role returns and called it terminal reward.

These historical scalar series are invalid as headline skill evidence. Preserve
the raw logs, but do not cite them as measurements or cosmetically perturb them.
The Elo formula/game payoff/opponent sampler are not changed by this correction.

## New measured quantities

- **Reward:** actual shaped strategy episode return, separately for each role,
  without arbitrary shift. It is not just the terminal TAG payoff.
- **Finite-window completion:** P completes by valid TAG; E completes by surviving
  tag-free to truncation. Every role episode is in the denominator, including
  physical failures. An opponent failure does not become a TAG or evader victory.
  This benchmark is distinct from the game's 0/0 timeout terminal reward.
- **Difficulty:** the empirical failure fraction of ONE declared frozen reference
  policy against the sampled opponent under matching reset/action/window settings.
  This challenge is fixed after calibration, not updated from learner outcomes.
- **Relative skill:** an online ability estimate compared to that reference,
  updated using all finite-window completions for calibrated opponent items.
  Sigmoid ability .5 denotes the reference ability, not 50% TAG probability.

The likelihood uses `P(completion) = sigmoid(ability - challenge_logit)` and
`challenge_logit = logit((reference_failures + .5)/(reference_episodes + 1))`.
One batch produces one bounded residual update, not thousands of optimizer steps
from correlated environments. The plot's difficulty uses the raw measured
failure fraction; the half-count smoothing only keeps model logits finite.
There is no extra assumed .65 role bias on top of the empirical reference.
Pursuer/evader abilities are independent. No combined skill/difficulty is defined;
the overview displays the two roles in separate panels.

This is a reference-relative online proxy, not absolute skill, a pristine test
set, an unbiased estimate under all nonstationary policies, or a guarantee of
generalization. Fixed-opponent checkpoint evaluations remain the stronger check.
An independently calibrated challenge can still be constant when the opponent
is unchanged or has the same measured challenge. Never force visible variation.

## Missing data and provenance

Recent/self-play, scripted and stationary opponents currently lack this independent
calibration. Their difficulty is a **gap**, not .5 or `1 - skill`. Ability updates
skip uncalibrated items; counts, availability and coverage are logged. Replaying
old logs uses the verified balanced-role batch layout and crosschecks terminal
causes against completion metrics. Failed or incompatible calibration protocols
must stop reconstruction rather than silently borrowing another distribution.

Calibration **4393**: historical 005/010/015/020, fixed reference
`tag_clean_start_mixed_4096_seed4_g4/checkpoints/000000000950`, 1.5 m, 10 seconds,
32 noise/reset seeds 7000–7031, both learner roles. Match each of heading noise
0 / +/-30 / +/-90 degrees with both-stochastic actions; add heading-0 stochastic
strategy/deterministic motor for the new motor-sampling comparison. These are
measurement jobs, not training or a change to the scheduled experiments.

The calibration reference is declared before reading those outcomes but was
previously used in development; do not call it an untouched test anchor.
File hashes, reference identities, calibration counts, reset/noise protocol and
source run IDs must accompany every corrected plot.

## Logging and corrected plots

New keys are `benchmark/{pursuer,evader}/relative_skill`, `opponent_difficulty`,
`observed_completion`, `calibration_coverage`, and availability/update counts.
`progress/{pursuer,evader,combined}` now uses those values, never legacy Elo.
When calibration is absent, plots explicitly say uncalibrated and show actual
return/completion only. Default training does not manufacture a reference table.
Set `benchmark_calibration_file` to the compiled, protocol-matched collection
to enable live estimates in future runs. The selected table is saved in the run
manifest. The logging change does not modify gradients, rewards or matching.

Existing running Python processes retain their already-imported callbacks. Their
original flat images cannot be fixed by changing this checkout alone; they need
explicit postprocessing. Rebuild old completed runs from immutable raw outcomes
after calibration. Publish labelled companion reanalysis runs rather than silently
rewriting training history. Keep fixed-reference initial/final evaluation counts
separate from the reconstructed online proxy.

## Dispatch and validation

Core rating tests (10), plot/trainer regressions (3), and replay/calibration
fixtures (5) pass, as do existing role-metric, Elo compatibility, renderer-role,
motor-sampling, reward/truncation and cadence checks. The first-record balanced
terminal-cause crosscheck also passed on all 32 old real runs. None of these CPU
checks substitutes for real calibration measurements.

GPU array **4393** is queued on g3 (16 tasks, maximum 8 GPUs). CPU-only job
**4394** depends on `afterok:4393`, then reconstructs and publishes the 32 old
runs automatically. The latter requests no GPU and explicitly forces CPU JAX.
Corrected W&B group: `tag_anchored_progress_reanalysis`, with source-run ID/URL,
raw-data hashes, reference calibration provenance and missing-data coverage.
There is no resume/overwrite of original training run IDs.

Manual reproduction after calibration completes:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/evaluation/rebuild_tag_progress.py \
  --calibration-dir logs/hierarchical/difficulty_calibration/job4393 --publish
```

Omit `--publish` for local-only output. Do not dispatch this again while 4394
is pending/running; it would duplicate companion W&B runs. Local output lives
under `logs/hierarchical/diagnostics/anchored_progress_reanalysis/`.
Its `calibration_collection.json` can be supplied via
`benchmark_calibration_file` in future training, provided the reset, motor and
duration protocol matches. Live logging alone cannot retroactively change
already-imported callbacks in the current 4385 processes; replay those after
they complete, explicitly selecting their run globs.

Status: implementation and CPU verification complete; calibration and corrected
W&B backfill are **queued, not yet complete**.
