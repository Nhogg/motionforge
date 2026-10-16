# MotionForge — mjlab self-play

This branch uses the smaller `mofo` implementation from `motionforge2` as the
default development path: a role-conditioned policy outputs joint actions
directly, trained with mjlab/rsl-rl PPO. The original hierarchical implementation
is retained as a reference, not the default trainer.

## Working checkout

```bash
cd /home/nhogg/motionforge
git branch --show-current  # mjlab-selfplay
export UV_PROJECT_ENVIRONMENT=.venv-mjlab
uv sync --locked
```

This is the original MotionForge checkout, switched to `mjlab-selfplay`.
`main` is unchanged. Its previous uncommitted changes, including untracked
scripts, are preserved in stash `c731bacb33ad7b8c0661028f660429ad7633e2eb`
(`main-wip-before-mjlab-selfplay-switch-2026-10-16`). `/home/nhogg/motionforge2`
is unchanged. The temporary extra worktree was removed.

The separate `.venv-mjlab` environment is inside this repository. Keep
`UV_PROJECT_ENVIRONMENT=.venv-mjlab` exported when using the new code, including
before submitting jobs, so the existing legacy `.venv` is not overwritten.
Original logs and checkpoints remain in place.

To return to the previous work, first save any new branch edits, then:

```bash
git switch main
git stash apply c731bacb33ad7b8c0661028f660429ad7633e2eb
```

`apply` keeps the saved stash available until you explicitly remove it.

## Entry points

```bash
# Inspect the fully resolved configuration without starting a run.
uv run --locked scripts/mofo_train.py --cfg job --resolve

# Environment/game checks (run GPU workloads through Slurm).
uv run --locked scripts/mofo_train.py check

# Evaluate a learned policy against another learned policy.
uv run --locked scripts/mofo_train.py eval \
  cmd.policy=/absolute/path/learner.pt \
  cmd.opponent_policy=/absolute/path/opponent.pt
```

Training entry point: `scripts/mofo_train.py tag`.
Configuration: `config/mofo_train.yaml`, with typed groups in `config/`.
Core task: `src/mofo/tasks/tag.py`; seat routing: `src/mofo/opponent.py`;
symmetric self-play: `src/mofo/selfplay.py`; league: `src/mofo/league.py`.

**Do not mistake the bare defaults for a self-play experiment.** Defaults have
`league.selfplay_frac=0`, snapshots disabled, no initial checkpoint, and a
zero-action opponent if neither `opponent` nor `cmd.init` is provided. Set the
intended policy initialization, opponents, self-play fraction and snapshots
explicitly before training. Example configuration, not an automatically queued run:

```bash
uv run --locked scripts/mofo_train.py tag \
  cmd.init=/absolute/path/stage1.pt \
  cmd.logdir=logs/mofo/selfplay_seed0 \
  seed=0 league.selfplay_frac=0.5 league.groups=8 \
  league.snapshot_every=200
```

Game defaults are distance-based TAG at 0.5 m, 20-second episodes, +/-10 outcome
reward, timeout as an evader win, and falls/OOB as losses. Contact TAG and shaping
are configurable. A win is not necessarily a valid TAG: report ending causes
separately. Compare initial and trained checkpoints under matched conditions.

## Slurm

Submit from this worktree, not the old checkout. The generic launcher uses
`SLURM_SUBMIT_DIR` and does not hard-code the old repository path.

```bash
mkdir -p logs/slurm
# Replace the checkpoint path and select your experiment settings first.
sbatch --nodelist=g4 scripts/slurm/prod.sbatch \
  uv run --locked scripts/mofo_train.py tag \
  cmd.init=/absolute/path/stage1.pt \
  cmd.logdir='logs/mofo/selfplay-{job}' \
  league.selfplay_frac=0.5 league.groups=8 league.snapshot_every=200
```

See [slurm.md](slurm.md) for cluster details and
[the import record](docs/mjlab_selfplay_import.md) for provenance and limitations.
