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
For evader-focused runs, set `cmd.render.learner_role=evader`; the learner is
then blue and its opponent remains the red pursuer. The default is pursuer.

Evaluation records which role fell or left the arena at the game endpoint and
during recovery. Recovery stops at the first failure, so the other robot's later
behavior is censored. `controlled_tag_rate_pursuer` defaults to requiring both
robots remain upright AND in bounds for the recovery window. For a separately
labelled fall-only audit, use `cmd.post_tag_ignore_oob=true` in evaluation:
pre-TAG arena exits still end the game, but post-TAG exits no longer end the
physical audit. Keep these two scores separate; JSON records
`recovery_failure_definition`.

Trial JSON also records learner path length, displacement, and own-seat motor
regularizer returns (discounted and undiscounted) through the game endpoint.
These exclude TAG, shaping, failure and contact-speed terms: they are diagnostics
of movement costs, not the total training/game return.

For a scripted-opponent curriculum, the training league supports
`league.scripted_speed_start`, `league.scripted_speed_end` and
`league.scripted_speed_ramp_iters`. These scale only the scripted opponent's
linear velocity commands, not yaw commands, rewards, physics frequency or
learned opponents. Defaults are full speed throughout. This is an opponent
command scale, not a guarantee of proportional physical running speed.
Evaluation and checkpoint renders stay full speed unless an easier scripted
audit explicitly sets `cmd.scripted_speed_scale`; JSON records the scale.
Never report slow-scripted training/evaluation wins as full-speed or self-play wins.

`rules.arena_observation=true` appends four TAG inputs: normalized arena-center
xy and the arena's +X axis xy, both in the robot's yaw frame. This provides wall
direction/orientation that scalar wall distance alone cannot identify. The six
legacy TAG features remain an unchanged prefix; the default is false. Hidden
layers and rewards are unchanged, but actor/critic input widths grow by four.
Use the checkpoint's saved observation config for evaluation. Frozen legacy TAG
opponents can ignore the appended inputs without modifying their weights;
loading a new-input policy into a legacy-input environment fails explicitly.

TAG-only reward ablation: `rules.upright_reward_offset=1` subtracts the upright
term's maximum reward rate from each learner seat. Perfect upright posture then
earns zero instead of a positive living bonus; tilt retains the same relative
penalty. The default is zero (original behavior). This changes episode-duration
incentives, so test it explicitly, not as a silent default change. TAG payoff,
other regularizers, and the standalone locomotion task are unchanged.

See [the shaping ablation protocol](docs/shaping_ablation.md) for TAG versus
failure semantics, hurling diagnostics, evaluation and the g4 Slurm array.

Local experiment results, decisions and job manifests are kept in
`docs/experiments.md` and `docs/experiments/`. These are intentionally Git-ignored;
only reusable code/configs and general documentation should be committed.

Resume a TAG checkpoint with `cmd.resume=/absolute/path/model_999.pt`,
`cmd.init=null`, an explicit `opponent=/absolute/path/STAGE1.pt`, and a new
`cmd.logdir`. Supply the same task/agent/league configuration as the source;
`cmd.iterations` specifies additional updates, and checkpoint numbering continues.
Actor, critic, normalization and optimizer state are restored; physics resets.
New checkpoints also save league members, outcome EMA and sampler RNG. Legacy
checkpoints rebuild the sibling `league/learner_*.pt` pool, but its unavailable
EMA/RNG restart. Opponents are freshly assigned to reset worlds on resume.
Use `cmd.resume_league=false` only for an intentional new opponent pool.

`agent.separate_roles=true` uses independent pursuit and evasion actor/critic
networks, normalization statistics and action-noise parameters. Both still use
the same observations, 29 joint actions, rewards and PPO implementation. This
roughly doubles network parameters; it does not introduce hierarchical control.
To initialize both roles exactly from a shared TAG checkpoint, explicitly set
`cmd.resume_optimizer=false` along with `cmd.resume`. This starts a fresh
optimizer; use the same setting in a matched shared-policy control. A checkpoint
already trained with separate roles restores its optimizer normally. Saved
checkpoints, frozen league snapshots, evaluation and videos retain role routing.
Training logs include `Role/pursuer/*` and `Role/evader/*` endpoint rates with
physical failures kept in the denominator. Self-play contributes two learner
seats per physical episode; evaluation still counts each physical world once.

Evader-weighted league training (`+experiment=evader_ladder`): `league.league_role=evader`
seats the learner as evader in every league-group world, while self-play worlds keep
`rules.learner_role`. `league.sampling=balanced` weights each member by `p(1 - p)` of the
learner's win EMA, favoring opponents it beats about half the time; with an evader-only
league that EMA is the evader's. `league.anchor_init_pool=true` keeps `init_pool`
checkpoints (e.g. a pursuer difficulty ladder of earlier snapshots) from eviction.
`agent.max_learning_rate` caps the KL-adaptive learning rate (rsl-rl allows up to 1e-2).
