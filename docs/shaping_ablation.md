# Clean TAG outcomes and shaping ablation

## Rules shared by every arm

- Pursuer wins only by a valid physical TAG; evader wins only by a healthy 20 s
  timeout without a tag. Game outcome reward remains +/-10.
- A fall/OOB by either agent ends the trial, censors the game outcome, and gives
  the offending agent an individual -10 penalty. Simultaneous failures penalize
  both seats. No opponent-fall/OOB bonus and no failure-as-TAG credit.
- Failures take priority over simultaneous contact/timeout. Do not interpret the
  absence of a tag in a failed trial as a successful evasion.
- `fall_loses` / `out_loses` now enable the corresponding **own** penalty; they
  do not award opponent wins. Keep both true for this experiment.
- `tag_mode=contact`, pursuer whole-body contact (`tag_bodies=[pelvis]`). Distance
  TAG remains available, but is not mixed into this experiment.
- Flat ground, the existing reset distribution, gamma .99, PPO and regularizers
  are held fixed. No anti-lunge penalty, terrain change or discount sweep yet.

## Matrix (one shared trainer and launcher)

`config/experiment/shaping.yaml` fixes common settings. Nine array tasks:

| Task IDs | Shaping | Seeds |
|---|---|---|
| 0–2 | disabled, weight 0 | 0, 1, 2 |
| 3–5 | potential, weight .25 | 0, 1, 2 |
| 6–8 | potential, weight 1 | 0, 1, 2 |

Each arm starts from the same explicitly supplied mjlab stage-1 checkpoint and
fresh optimizer. 4096 physical worlds, 24 steps/update, 1000 updates (98.304M
world transitions; 50% symmetric self-play adds learner seats, not more worlds).
The remaining worlds use 8 PFSP groups, with snapshots every 100 updates. Failed
trials are excluded from game-win matchmaking updates, not counted as draws/wins.

## Diagnostics and evaluation

The existing metrics manager logs valid TAG, healthy timeout and physical
failure separately; exact episode sums for TAG/shaping/failure rewards; and
terminal distance, relative closing speed, pursuer root tilt/angular speed and
peak inter-agent force over the last control step's contact history. Force is
not reported as impulse. Existing regularizer logs remain available.

`scripts/mofo_train.py eval` now records one first episode per physical world,
including the actual terminal state (no post-reset measurement). It continues
physics for `cmd.post_tag_seconds=1.0` after a valid TAG. Falling/OOB during that
window fails the separate post-tag-survival measurement, but does not rewrite the
already recorded TAG outcome. Video includes the continuation. This changes no
training reward or termination rule. `controlled_tag_rate_pursuer` means a TAG
with both agents surviving that recovery window, not a force/tilt safety certificate.

The launcher evaluates `initial.pt` (before any PPO update), iterations 250/500
and final 999, against the same scripted locomotion reference and against the
checkpoint itself. Evaluation seed is always 30000, 256 first episodes per
condition, deterministic actors, no observation corruption. Random role counts
and Wilson intervals are saved; do not count two learner seats as independent
episodes. Post-tag force/speed/tilt and survival are reported conditional on tags;
the controlled-tag rate uses all pursuer trials. No tags => conditional metrics
are null, never zero or 100% survival. All three training seeds stay separate.

Outputs: `logs/mofo/shaping_<array>/w<weight>_s<seed>/`; TensorBoard training
logs, resolved Hydra config, `eval/model_<iteration>.json` for scripted cross-play,
`eval/selfplay/model_<iteration>.json` for learned self-play, and `initial.json`
in each evaluation directory. Raw per-trial rows are embedded in the JSON.
Seed 0 produces a predefined final self-play MP4 per arm; no success-video search.

Select settings by valid TAG and controlled-tag rates, evader healthy completion,
failure rates and contact quality across seeds. Shaped return is not a fair
cross-weight selection metric. Follow a promising setting with fixed, held-out
cross-play before adding an anti-lunge reward or harder terrain.

## Run on g4

```bash
cd ~/motionforge
export UV_PROJECT_ENVIRONMENT=.venv-mjlab
uv sync --locked
mkdir -p logs/slurm
# Replace with the real stage-1 .pt checkpoint, not an old JAX hierarchy checkpoint.
sbatch --job-name=mofo-shaping --nodelist=g4 --array=0-8%8 \
  --cpus-per-task=16 --mem=48G --time=04:00:00 \
  --output='logs/slurm/%x-%A_%a.log' \
  scripts/slurm/prod.sbatch bash scripts/slurm/shaping_ablation.sh \
  /absolute/path/stage1.pt
```

All runs use the shared `scripts/slurm/prod.sbatch` template. The sweep worker
requires a real absolute checkpoint path; it will not silently substitute the
zero-action opponent. These submission flags target g4, eight concurrent one-GPU jobs.
An existing allocation holding all eight g4 GPUs must finish/release them first.
No unrelated allocation is cancelled by this launcher.

First verify a small end-to-end run of each condition:

```bash
sbatch --job-name=mofo-shaping-smoke --nodelist=g4 --array=0,3,6 \
  --cpus-per-task=16 --mem=48G --time=00:15:00 \
  --output='logs/slurm/%x-%A_%a.log' --export=ALL,SMOKE=1 \
  scripts/slurm/prod.sbatch bash scripts/slurm/shaping_ablation.sh \
  /absolute/path/stage1.pt
```

Temporary feature-test scripts were removed after validation at the user's request.
Existing GPU `check` exercises contact isolation, reset/shaping telescoping,
symmetric seat rewards and league routing; it remains the common integration test.

## Verification before the real checkpoint is supplied

- Eight CPU reward/recovery regressions and the existing ten league checks pass.
- GPU integration check **4850** passes, including mixed touching/non-touching
  worlds, 32,768 reset samples, shaping terminal/reset checks, symmetric seat
  rewards, and failure exclusion from PFSP records.
- GPU PPO smoke **4847** saves the pre-update `initial.pt` and the two-update
  checkpoint with finite training metrics and both learner seats active.
- GPU evaluation **4849** validates one first episode per world, explicit manual
  resets, frozen TAG outcomes and a continuing post-TAG recovery window.
- GPU video **4851** produces a decodable MP4 with no EGL requirement.

These are software checks using an uninitialized smoke policy, not locomotion
warm-start validation or evidence that a shaping condition works. The actual
stage-1 checkpoint and the three-condition warm-start smoke remain prerequisites
for the production sweep. g4's existing interactive allocation is not modified.
