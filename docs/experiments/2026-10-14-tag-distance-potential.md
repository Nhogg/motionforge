# TAG distance-potential shaping (v2)

This replaces the independently clipped .02-per-strategy-step progress term.
Array 4296 was canceled at the user's request; its logs/checkpoints are retained.
The replacement starts fresh strategies with the same pretrained frozen P2
locomotion and otherwise unchanged diagnostic settings.

## Definition

```text
Phi_P(s) = -alpha * planar_distance(s) / D
Phi_E(s) = -Phi_P(s)
shape_P  = strategy_gamma * Phi_P(next_state) - Phi_P(previous_state)
shape_E  = -shape_P
```

Default `alpha=.25`. D is fixed from arena geometry, never episode distance:
the arena is an 8m by 8m square, whose corner-to-corner diameter is
`2 * sqrt(2) * arena_half_extent = 11.3137085m`. An explicit positive
normalization overrides this default.

```text
tag_reward.progress.enabled=true
tag_reward.progress.alpha=0.25
tag_reward.progress.distance_normalization_m=null
```

The old `progress.weight` and `progress.distance_scale_m` keys are removed.
Both trainer paths pass the very same discount from their strategy GAE
configuration into rollout reward construction. Standalone collection defaults
to .99, matching standalone GAE's default; callers overriding GAE must also
supply `strategy_discount`. The diagnostic/training CLI keeps its existing
strategy discount 1.0. The locomotion discount and objective are untouched.

Both role rewards use the same world-space distance between the two agents,
not separately rotated heading-frame distances. Shaping is evaluated at
strategy-interval endpoints, including the actual endpoint of partial intervals.
There is no clipping or saturation of normal progress. Simulator NONFINITE
endings remain explicitly classified/excluded; finite state jumps remain
visible in `diagnostics/max_control_distance_jump_m` and
`diagnostics/max_strategy_distance_jump_m`. Non-finite training metrics are
recorded and then stop training rather than silently substituting capped values.

The geometric potential is evaluated at recorded physical endpoints, including
TAG/failure endpoints, exactly as specified. This patch does not introduce an
extra absorbing-terminal potential convention or claim an additional policy-
invariance theorem for the finite terminated game.

## Why oscillation cannot farm net shaping at the current discount

For gamma=1:

```text
sum(shape_P) = alpha * (initial_distance - final_distance) / D
```

Returning to the same distance gives zero net shaping regardless of the number
of toward/away cycles. For in-arena endpoints, the net magnitude is at most
alpha=.25, not `.02 * number_of_steps`. Absolute reward activity (sum of absolute
increments) may still be large during oscillation; it is not net free reward.
OOB/pathological state jumps can exceed the geometric range and are diagnosed,
not silently clipped.

For gamma<1, the DISCOUNTED sum telescopes to
`gamma**T * Phi(final) - Phi(initial)`. Individual closer/farther sign guarantees
are specific to gamma=1: with the specified negative distance potential and
gamma<1, a sufficiently small distance increase can still have positive pursuer
shaping. Tests check the exact formula and discounted identity rather than
silently changing gamma or the potential to force those signs.

## Preserved behavior and logging

Valid TAG component remains exactly +1 pursuer / -1 evader. Timeout terminal
TAG component remains 0/0, with next-state value bootstrap. True termination
still zeroes bootstrap. Failure penalties, command smoothness, symmetric
boundary penalty/projection, architecture, role encodings, stop-gradient,
locomotion reward, frequencies, optimizer schedules and opponent mixture stay
unchanged. Component balance still needs observation; do not assume the
unchanged smoothness penalty is negligible relative to smaller shaping.

Existing `reward/{role}/progress` fields are per-strategy-step means; episode
signed sums, absolute sums and maxima are logged separately. Actual component
values at strategy endpoints are in transitions; per-environment component
episode sums are retained in training JSONL. Training manifests identify
`strategy_reward_version=tag_distance_potential_truncation_v2` and save resolved
alpha, D and strategy gamma. Evaluation inherits enabled/disabled status and
the potential configuration from the checkpoint's run manifest, rather than
turning progress on accidentally for the control. Old explicitly clipped
reward configurations are rejected rather than relabeled as potential shaping.
Checkpoints predating strategy shaping use documented current defaults for
evaluation reward diagnostics; policy/tag outcomes remain measured separately.

## Verification and replacement experiment

CPU tests cover positive closing/negative retreat at gamma=1, exact role
antisymmetry, no clipping, explicit fixed normalization, disabled shaping,
oscillation cancellation, discounted telescoping, unchanged TAG component and
truncation bootstrap. Synthetic JAX collector tests cover interval endpoints,
partial-interval cutoff retention and episode diagnostics.

GPU suite: job 4305, `scripts/slurm/hierarchical_tag_reward_smoke_g4.sbatch`.
It also checks both trainer bootstrap paths, actual potential telescoping,
frozen locomotion parameters/optimizer, balanced PPO, videos and end-to-end CLI
training/evaluation. A worker reading mixed transition schemas during active
edits caused an earlier attempt to fail; the final suite force-refreshes project
bytecode before testing a consistent source version.
Job 4305 passed the complete GPU suite, including CLI training and saved-config
evaluation. Evaluation confirms alpha=.25, D=11.313708498984761m, gamma=1.0 and
the v2 protocol loaded from the checkpoint's run manifest. All 12 CPU regression
tests and the synthetic collector tests pass; Ruff and shell syntax checks pass.

Replacement launcher: `scripts/slurm/hierarchical_tag_reward_diagnostic_g4.sbatch`.
Four seeds, each paired with potential enabled/disabled: eight one-GPU g4 tasks.
Alpha=.25 and D=arena diameter in the enabled arm; both arms retain identical
truncation/failure/smoothness/boundary semantics and the original historical
opponent mixture. Fresh strategy initialization, frozen known-good locomotion,
100 iterations / 204.8M steps, with matched before/after evaluation.
W&B group: `hierarchical_tag_potential_diagnostic`.

Replacement array **4306** was submitted after the successful suite. Array 4296
is canceled; its outputs are retained and must not be pooled with v2 results.
