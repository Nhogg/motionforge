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
