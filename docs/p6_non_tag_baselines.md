# P6 non-TAG baselines

P6 constructs controlled datasets for comparison with future TAG trajectories.
Every dataset budget is measured in valid single-agent control timesteps rather
than episodes, so early termination cannot silently reduce one condition's data
volume. Seeds, configuration, checkpoint provenance, schema version, and an
output SHA-256 digest are written to a JSON manifest.

## Ordinary random velocity commands

The first baseline uses one G1 on flat terrain with no opponent and no external
pushes. It restores the accepted P2 locomotion checkpoint and deliberately
reuses `G1StandingJoystick`'s canonical structured command distribution. This
is the ordinary distribution on which the controller was trained: standing,
pure forward/lateral/yaw commands, and mixed commands sampled within the
configured velocity limits. P6 does not alter it to manufacture hard cases.

`motionforge/datasets/locomotion.py` defines a versioned per-agent raw sample
boundary independent of policy inference, physics stepping, and serialization.
`scripts/datasets/generate_random_commands.py` owns deterministic episode
resets and writes exactly the requested number of JSONL rows plus a manifest.
Opponent and game fields are absent rather than represented by misleading
zeros. Episode and timestep identifiers preserve derivative boundaries for
later P10 analysis.

The initial smoke budget is intentionally small. It validates the complete GPU
checkpoint-to-dataset path before generating the final equal-sized P6 datasets.
The manifest hash identifies the exact artifact; it is not treated as a promise
of byte-identical MJX-Warp floating-point trajectories across separate process
runs. Seeds and command schedules are reproducible, while long closed-loop GPU
rollouts may diverge numerically from small kernel-order differences.

Evidence: `logs/p6/random_commands_1024_d_manifest.json`. The GPU audit wrote
exactly 1,024 schema-v1 samples, remained in one episode, and crossed two
ordinary command transitions. All manifest checks passed: checkpoint loading,
finite values, flat terrain, disabled pushes, absence of opponent fields,
consistent schema, and an upright initial state. This is validation evidence,
not yet the final equal-budget P6 dataset.

## Deliberately aggressive command sampling

The aggressive baseline uses the same checkpoint, flat environment, raw sample
schema, exact timestep budget, and disabled-push setting as the ordinary
baseline. Only its high-level command schedule changes. The schedule cycles
through paired phases for hard turns, forward/reverse motion, braking, lateral
reversal, and high-yaw reversal. A seed rotates the maneuver order and selects
the initial direction without allowing any required maneuver to disappear.

Each JSONL row records its maneuver name and phase. The manifest reports counts
for every maneuver and phase, making coverage an explicit acceptance check
rather than a probabilistic consequence of random sampling. Commands remain
inside the accepted locomotion controller's trained limits.

Evidence: `logs/p6/aggressive_commands_1024_a_manifest.json`. The GPU audit
wrote exactly 1,024 schema-v1 samples in one episode with 40 command
transitions. Every required maneuver and both phases were present, including
high yaw rates, and all state, controller-range, environment, and provenance
checks passed. As with the ordinary audit, this validates the generator but is
not yet the final equal-budget dataset.

## Terrain curriculum without an opponent

The terrain baseline uses the ordinary structured command sampler and the same
accepted checkpoint and per-agent schema as the other P6 conditions. It adds
no opponent and no external pushes. Fixed-length collection episodes progress
through rough-heightfield probabilities `[0.0, 0.5, 0.75, 1.0]`, so early data
is flat-only and the final stage is rough-only. Intermediate terrain choices
are seeded and recorded per row.

The implementation reuses Playground's existing G1 flat plane and rough
heightfield models. It does not introduce the slopes, bumps, gaps, or
adversarial terrain reserved for later phases. The manifest reports sample
counts by curriculum stage and terrain type, plus early physical terminations.

Evidence: `logs/p6/terrain_curriculum_1024_a_manifest.json`. The GPU audit
wrote exactly 1,024 schema-v1 samples across all four stages, split evenly
between flat and rough terrain. All environment, command-range, schema,
provenance, and opponent-absence checks passed. Two episodes terminated early
and were reset deterministically; the exact sample budget was preserved across
nine actual episodes instead of silently discarding those failures.
