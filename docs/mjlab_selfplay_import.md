# mjlab self-play branch import

- Branch: `mjlab-selfplay` in the original MotionForge repository.
- Worktree: `/home/nhogg/motionforge-mjlab-selfplay`.
- Original committed base: `d3640df`.
- Imported repository: `/home/nhogg/motionforge2`.
- Imported commit: `7d11a7ebfd64c1dde6d66121f1dd430c80594a5c`.

The merge preserves the original committed tree, overlays the tracked files from
the imported commit, and records its history as a merge parent. In particular,
pruning commits in the smaller repository do not delete the legacy implementation
here. Imported Python code and experiment defaults are unchanged. README and
ignore rules are adapted for this combined checkout.

The new `pyproject.toml` and `uv.lock` define the default mjlab environment and
package both `mofo` and the retained `motionforge` tree. Legacy training may need
dependencies from the original environment; keeping its files is not a claim
that every historical entry point runs under the new dependency set. Use the
original checkout/environment for legacy experiments.

The original checkout's dirty worktree was not imported, stashed, reset or
committed. Those changes remain at `/home/nhogg/motionforge`. No running jobs were
changed, and no training was started as part of this import. No external push was
performed. Existing logs, checkpoints and virtual environments remain in their
original locations; use explicit checkpoint paths from this worktree.

Scientific interpretation remains unchanged by importing the code: the direct
joint-action policy is not the old hierarchical method. Symmetric self-play and
PFSP are supported, but disabled/not exercised by several bare defaults. Claims
require inspecting actual resolved run configs and separate TAG/fall/OOB/timeout
results, not treating total wins as tags.

## Import validation

- `uv sync --locked`: successful; separate `.venv` created in this worktree.
- `uv run --locked scripts/mofo_train.py --cfg job --resolve`: successful.
- `uv run --locked python -m mofo.league_test`: all 10 checks passed.
- Generic debug/production Slurm scripts pass `bash -n`.
- Imported source, config, launchers, dependency manifest and lockfile match the
  source commit exactly. Legacy source paths are unchanged from committed main.
- Original checkout diff and untracked-file hashes match their pre-import values.

No GPU simulation or training was launched for this import; these checks establish
environment/CLI/league integration, not task performance or GPU validation.
