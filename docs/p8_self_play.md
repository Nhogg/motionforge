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
