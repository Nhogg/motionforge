# W&B trajectory integration and next TAG experiment

## Published diagnostics

Eight checkpoint trajectory plots and the corrected role-labeled video were
successfully synced to the diagnostic W&B run:
https://wandb.ai/luc-ssl/MotionForge/runs/ucorfjaz

Publisher: `scripts/evaluation/publish_hierarchical_tag_audit.py`.
It creates a separate diagnostic run; it does not resume or alter training runs.

The training video callback now saves PNG/SVG plots and raw trajectory JSON from
the exact simulation used for each video. W&B keys are
`evaluation/pursuer_trajectory` and `evaluation/evader_trajectory`, alongside the
existing role video keys. Additional single-rollout scalars report tagging,
pursuer out-of-bounds, final separation, pursuer approach speed, and retreat fraction.
These are video-rollout diagnostics, not estimates of population success rate.
Green is always pursuer; orange is always evader. The updated callback passed its
GPU regression test in Slurm job 4270, including role/head/color routing and
same-rollout plot samples. Already-running processes will not hot-load this change.

## Why test opponent coverage next?

The old mixed league is 75% historical and divides the remaining 25% equally
among current policy, stationary, and scripted opponents. Current-policy
self-play therefore appears in only 8.33% of training episodes.

Seed 1 from mixed-frozen diagnostics succeeds against historical opponents and
independent trained opponents, but the audited self-play trajectories retreat
and exit bounds without tagging. Correct role routing is verified: this is a
real behavioral failure, not both agents being rendered as evaders.
Opponent coverage is a plausible hypothesis, not an established cause.
At iteration 100 of job 4234, last-20-iteration mixed-training pursuer tag rates
are 70% (seed 4), 95% (seed 5), and 0% (seed 2 and seed-0 control).
Seed-2 and seed-0 control out-of-bounds rates are 84.7% and 68.8% respectively.
Longer-credit (GAE=1) has not rescued seed 0: 0% tags, though 0% out-of-bounds.
These mixture-dependent training windows are not held-out evaluation scores.

## Matched continuation experiment

Launcher: `scripts/slurm/hierarchical_tag_recent_coverage_g4.sbatch`.
Eight array tasks, each using one GPU on g4. Four starting checkpoints at
iteration 100: diagnostic mixed-frozen seed 1 (job 4199), and follow-up fresh
seeds 2, 4, 5 (job 4234). Each checkpoint supplies both arms:

| Arm | Historical | Current self-play | Stationary | Scripted |
| --- | ---: | ---: | ---: | ---: |
| Control | 75% | 8.33% | 8.33% | 8.33% |
| Recent-heavy | 25% | 50% | 12.5% | 12.5% |

This compares mixture recipes, not a pure current-policy-only effect: historical
and curriculum shares also change. All other settings are matched: frozen
pretrained locomotion, restored strategy and optimizer, GAE=.95, learning rate
3e-4, fixed 1.5m separation, 4096 environments, horizon 500, and 10s episodes.
Each arm adds 100 iterations (204.8M steps), ending at checkpoint 200.
Initial checkpoint 100 is copied into the new run for a reproducible baseline.
W&B videos and paired plots are emitted every 51.2M steps.

At checkpoints 100 and 200, the script evaluates both roles with deterministic
and stochastic actions: 32 reset seeds against current self-play, and 16 seeds
against historical checkpoints 005 and 020. Reset seeds start at 2000, fixed
separation is 1.5m, and episode duration is 10s. Self-play changes both agents as
training progresses, so historical cross-play supplies a fixed-opponent anchor.
Evaluation JSON lives under each new run's `evaluation/iter_*/` directory.
The evaluator's new self-play option has a short GPU smoke test queued as
job 4277; it was resource-blocked by g3's full-node allocation at handoff.
CPU mixture and renderer-role tests pass, and source update configurations
were verified identical and compatible with the continuation launcher.

Submit from the login host after creating the Slurm output directory:

```bash
cd /home/nhogg/motionforge
mkdir -p logs/slurm
ssh head0 'cd /home/nhogg/motionforge && sbatch --dependency=afterok:4234 scripts/slurm/hierarchical_tag_recent_coverage_g4.sbatch'
```

This queues behind the full job 4234 array (including its evaluations), and
blocks if any task fails. Inspect failed dependencies instead of removing them
blindly. The new training array has not been submitted by Codex.

## Decision after the run

Compare paired changes in actual pursuer tag rate, physical survival, and
trajectory approach/retreat behavior. Retain the recent-heavy recipe only if
self-play improves without losing historical-opponent competence across
multiple seeds. Single video success is insufficient. Elo is reinitialized
for continuations, so do not compare its absolute level across stages.

If this works, repeat the selected recipe from random strategy initialization
with the disclosed pretrained P2 controller, multiple seeds, and held-out
opponents. Then expand separation from fixed 1.5m to 1.5–3m as a separate
curriculum test. These continuations are tuning evidence, not the final
from-initialization publication result.
