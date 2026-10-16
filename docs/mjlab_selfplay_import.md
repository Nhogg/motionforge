# mjlab self-play branch import

- Branch: `mjlab-selfplay` in the original MotionForge repository.
- Working checkout: `/home/nhogg/motionforge`.
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

The initial import was prepared in a temporary worktree. At the user's direction,
the original checkout was then switched to this branch and that temporary
worktree removed. The original dirty main changes were saved with untracked files
in stash `c731bacb33ad7b8c0661028f660429ad7633e2eb`, named
`main-wip-before-mjlab-selfplay-switch-2026-10-16`. No user edits were committed
or discarded. `main` itself remains at the original commit.

No running jobs were changed and no training was started. No external push was
performed. Existing logs, checkpoints and the legacy `.venv` remain in place.
Use `UV_PROJECT_ENVIRONMENT=.venv-mjlab` for the imported implementation, so
installing its dependencies does not overwrite the legacy environment.

Scientific interpretation remains unchanged by importing the code: the direct
joint-action policy is not the old hierarchical method. Symmetric self-play and
PFSP are supported, but disabled/not exercised by several bare defaults. Claims
require inspecting actual resolved run configs and separate TAG/fall/OOB/timeout
results, not treating total wins as tags.

## Import validation

- `UV_PROJECT_ENVIRONMENT=.venv-mjlab uv sync --locked`: successful in the
  original checkout; the old `.venv` was left intact.
- `uv run --locked scripts/mofo_train.py --cfg job --resolve`: successful.
- `uv run --locked python -m mofo.league_test`: all 10 checks passed.
- Generic debug/production Slurm scripts pass `bash -n`.
- Imported source, config, launchers, dependency manifest and lockfile match the
  source commit exactly. Legacy source paths are unchanged from committed main.
- Original checkout diff and untracked-file hashes match their pre-import values.

No GPU simulation or training was launched for this import; these checks establish
environment/CLI/league integration, not task performance or GPU validation.
