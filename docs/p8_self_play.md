# P8 self-play

## Frozen learned evader integration

The first P8 change allows the pursuer environment and its existing PPO
launcher to use either the accepted scripted evader or a frozen learned
evader. Leaving `evader_checkpoint=null` preserves the P7 scripted-opponent
behavior. Supplying a checkpoint restores `FrozenLearnedEvader` and records
its immutable checkpoint-backed specification through the launcher's existing
manifest fields.

The learned evader consumes the same normalized nine-value observation used
by the TAG policies. Its previous high-level command is held in
`TagPursuerPipelineState`, updated on every locomotion-control step, and reset
to zero at episode reset. Keeping that memory in the JAX pipeline state makes
batched execution, JIT compilation, and autoreset behavior reproducible.

Example Hydra override:

```bash
uv run scripts/training/train_tag_pursuer.py \
  evader_checkpoint=logs/p7/training/tag_evader_fixed_noise_1m_seed0_retry/checkpoints/000000327680 \
  evader_fixed_noise_std=0.2 \
  output_dir=logs/p8/training/tag_pursuer_vs_learned_evader_smoke_seed0
```

This is an alternating frozen-opponent training primitive, not complete
self-play. Historical snapshot creation, opponent-population sampling, and
population win-rate evaluation remain later P8 objectives.

## Historical policy snapshots

`scripts/self_play/save_tag_policy_snapshot.py` copies a trained checkpoint
into `checkpoints/<role>_g<generation>_<digest-prefix>` beneath a population
directory. `population.json` records the complete digest, role, generation,
seed, fixed policy noise, source checkpoint, and relative copied checkpoint.
The command rejects an existing role/generation and verifies the copied digest
before publishing it, so a later training run cannot mutate historical policy
evidence through a shared path.

The initial accepted P7 pursuer and evader can be registered as generation
zero. This registry deliberately does not yet select opponents; sampling is
the next independent P8 component.

Generation-zero evidence is stored in `logs/p8/population/population.json`.
It contains copied and digest-verified versions of the accepted P7 pursuer and
evader checkpoints. The implementation audit is recorded at
`logs/p8/tag_policy_population_a.json`.

## Deterministic opponent selection

`select_opponent` chooses between an explicit current checkpoint and the
historical snapshots for the requested opponent role. `current_probability`
controls the current-versus-history draw; a second deterministic draw selects
within the generation-sorted historical pool. Both draws are derived from a
SHA-256 hash of the role, current generation, and experiment seed rather than
process-global random state.

The returned record contains the exact checkpoint, digest, generation, policy
noise, source category, selection seed, and snapshot ID. The standalone Hydra
command `scripts/self_play/select_tag_opponent.py` writes that record to JSON.
Historical checkpoint contents are rehashed against the registry before a
selection is returned.
Selection currently occurs once per training job; per-environment policy
mixtures are intentionally deferred until job-level alternating self-play is
validated.

## Alternating round runner

`scripts/self_play/run_tag_self_play_round.py` owns one sequential round. It
selects an evader, resumes the current pursuer checkpoint, trains and
snapshots the next pursuer generation, then selects a pursuer, resumes the
current evader checkpoint, and trains and snapshots the next evader
generation. The second selection may therefore include the pursuer produced
earlier in the same round.

Each round writes `round_manifest.json` before executing work and after every
completed stage. It records exact subprocess arguments, opponent selections,
result checkpoints, and snapshots. `dry_run=true` validates inputs and emits
the pursuer half of the plan without modifying the population or launching
training. Training launchers now expose `restore_checkpoint` explicitly so
self-play updates an existing policy rather than repeatedly training fresh
agents.

The first end-to-end smoke round is recorded in
`logs/p8/rounds/round_0001_smoke_a/round_manifest.json`. Each role completed
40,960 environment steps and produced a digest-verified generation-one
snapshot. The four-environment evaluation remained strongly pursuer-favored:
the pursuer tagged every evader, and the evader timed out in none of its four
episodes. This validates alternating training and artifact flow, but it is not
evidence that flat-ground self-play is balanced or stable.

## Population evaluation and rendering

`scripts/evaluation/evaluate_tag_population.py` evaluates the Cartesian
product of registered pursuer and evader snapshots on identical reset seeds.
It records terminal causes, pursuer and evader win rates, snapshot identities,
and generation indices for every matchup. Tags and evader failures count as
pursuer wins; timeouts and pursuer failures count as evader wins.

`scripts/evaluation/render_tag_learned_matchup.py` restores one learned policy
for each role and renders their integrated rollout to MP4 with a JSON sidecar.
The pursuer is red, the evader is blue, and frames are captured at the 10 Hz
high-level policy rate.

The first matrix is stored at `logs/p8/tag_population_matrix_a.json` and uses
four held-out seeds for each of the four generation-zero/generation-one
matchups. Generation-one versus generation-one ended in four tags and zero
timeouts. The corresponding seed-1000 video is
`logs/p8/videos/tag_generation1_seed1000.mp4`; it ends in a tag at 2.9 seconds
without either agent falling or leaving the arena.

### Learned-opponent control-rate correction

The first generation-one smoke round and matrix exposed a control-rate error
in the newly added learned-evader path. `TagPursuerEnvironment` evaluated the
evader policy inside each of five 50 Hz locomotion substeps, although that
policy was trained and observed at the 10 Hz high-level rate. The original P7
evader evaluator held each command for all five substeps. The environment now
computes one learned evader command per high-level step, holds it through the
locomotion scan, and records that exact command as policy memory. A regression
check compares the held command against direct deterministic inference.

Consequently, `round_0001_smoke_a`, `tag_population_matrix_a.json`, and its
first video are retained as debugging evidence but must not be interpreted as
valid self-play performance. A corrected matrix and render supersede them.

The corrected four-seed matrix is
`logs/p8/tag_population_matrix_control_rate_fixed_a.json`. Generation-zero
evader versus either pursuer timed out all four episodes. Generation-one
evader versus generation-zero pursuer was tagged in all four episodes, while
generation-one versus generation-one produced three timeouts and one tag.
Thus the corrected current matchup is evader-favored (75% versus 25%), but
both outcomes occur and the original accepted evader behavior is recovered.
No reward change was made because the apparent 0% evasion rate was an
execution bug rather than evidence that the evader objective was inadequate.

The superseding rollout is
`logs/p8/videos/tag_generation1_control_rate_fixed_seed1000.mp4`. It records a
complete 20-second evasion with no fall or boundary exit.

## Corrected generation-two round

The first full round after the learned-opponent control-rate correction is
recorded in
`logs/p8/rounds/round_0002_corrected_a/round_manifest.json`. Starting from the
corrected generation-one snapshots, both roles completed 143,360 environment
steps with finite metrics. Deterministic historical-opponent sampling selected
generation-zero evader `evader_g0000_a227dbd40b05` for the pursuer update and
generation-zero pursuer `pursuer_g0000_976aeb081d66` for the evader update.
The resulting digest-verified snapshots are
`pursuer_g0002_8415fb5f6f0c` and `evader_g0002_135f9fec082c`.

The corresponding held-out population matrix is
`logs/p8/tag_population_matrix_generation2_a.json`. It evaluates all nine
generation-zero through generation-two pairings on seeds 1000--1003. The
pursuer win rates, with rows denoting pursuer generation and columns denoting
evader generation, are:

| pursuer / evader | generation 0 | generation 1 | generation 2 |
| --- | ---: | ---: | ---: |
| generation 0 | 0% | 25% | 25% |
| generation 1 | 0% | 25% | 25% |
| generation 2 | 100% | 100% | 50% |

The generation-two current matchup is balanced on this small held-out set:
two tags and two timeouts. The matrix also provides a clear cycling signal.
The generation-two pursuer defeats both older evaders, while the newer
evaders retain a strong advantage over both older pursuers. This completes the
initial cycling/forgetting diagnostic, but four seeds per pairing are not
enough to distinguish catastrophic forgetting from seed variance or ordinary
non-transitive improvement. Flat-ground self-play therefore remains under
evaluation; difficult terrain is not introduced yet.
