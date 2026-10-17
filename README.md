# MotionForge

Direct joint-action humanoid TAG using mjlab and rsl-rl, on branch `mjlab-selfplay`.

```bash
export UV_PROJECT_ENVIRONMENT=.venv-mjlab
uv sync --locked
```

Main entry point: `scripts/mofo_train.py tag|eval|check`. Shared Hydra configs
are in `config/`; all reward ablations use the same trainer.

## Weights & Biases

Authenticate once on the login node with `uv run wandb login`, or export
`WANDB_API_KEY` before submission (never put the key in a CLI override).
Add these overrides to any training command:

```bash
cmd.logger=wandb cmd.wandb.project=MotionForge cmd.wandb.group=shaping_ablation
```

Optional: `cmd.wandb.entity=<team>` or `cmd.wandb.mode=offline` for disconnected
jobs. The W&B run name is the final directory name of `cmd.logdir`; give each run
a unique log directory. Full resolved task/PPO config, reward components, TAG and
failure diagnostics, league metrics and losses are logged. The x-axis is PPO
iteration, not physical-world steps (self-play adds learner seats). Local
TensorBoard events and checkpoints remain available; checkpoint upload is opt-in
with `cmd.wandb.upload_model=true`. Evaluation JSON/videos are not automatically
uploaded by this training logger unless checkpoint rendering is enabled (below).
Existing running jobs are not modified.

## Checkpoint videos

`cmd.render.enabled=true` (default) renders each saved `model_<iteration>.pt`,
including the final checkpoint, using the existing evaluator. Set checkpoint
frequency with `agent.save_interval=250`. Each checkpoint produces
`<logdir>/videos/model_<iteration>.mp4`, trial JSON, resolved evaluation YAML and
a renderer log. With W&B logging enabled, the MP4 appears under `video` at that
checkpoint's PPO iteration. `initial.pt` is not rendered.

The default is deterministic same-checkpoint self-play, fixed seed 30000, one
first episode, red pursuer/blue evader, and one second of post-TAG recovery.
These clips are diagnostics, not success-rate estimates or success-selected
examples. Use `cmd.render.matchup=scripted` for a fixed stage-1 opponent or
`cmd.render.enabled=false` to disable. Training pauses during rendering in a
separate process on the allocated GPU; this adds time and some GPU memory, but
does not step/reset the training worlds or consume the training RNG. Render
failures log `Render/success=0` and a warning; use
`cmd.render.fail_on_error=true` to make them fatal for a smoke test.

See [the shaping ablation protocol](docs/shaping_ablation.md) for TAG versus
failure semantics, hurling diagnostics, evaluation and the g4 Slurm array.
