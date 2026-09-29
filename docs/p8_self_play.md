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
