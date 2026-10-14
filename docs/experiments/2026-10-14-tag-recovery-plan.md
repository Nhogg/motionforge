# TAG recovery plan — October 14, 2026

All times America/Chicago (CDT). Working end-of-day deadline: 18:00.

Goal: identify a reproducible pursuit recipe and collect initial confirmation
evidence today. Success is an experimental outcome, not a promised deadline.
Publication readiness also requires fair baselines, sufficient seed variability,
and evaluations supporting the precise claims.

## Current experiment

Slurm array 4199 has eight runs: stationary/mixed opponents crossed with
joint/frozen locomotion, each at training seeds 0 and 1. All start with random
strategy parameters and pretrained P2 locomotion, fixed 1.5 m separation,
4096 environments, 100 iterations (204.8M environment steps). This is the
meaning of a fresh run here; it is not motor learning from random weights.

At approximately 10:23 all runs had reached iteration 11 (22.5M steps),
with approximately 55k steps/sec and finite metrics. Frozen runs had zero
locomotion updates. Training should finish around 11:20, with additional
time for four evaluations per run. This estimate can change with video and
evaluation costs; monitor actual progress.

Inputs: logs/slurm/tag-pursuit-diag-4199_*.{out,err} and
logs/hierarchical/training/tag_pursuit_diag_*_job4199/.
Monitor report: logs/hierarchical/diagnostics/job4199/status.md.

Historical opponents 005 and 020 in these evaluations are development probes:
the mixed league trains against them. They are not held-out opponents.
Evaluation JSON `passed` means execution/finite checks passed, not successful TAG.
Elo's 0.65 baseline is rating calibration, not a learning target.

## Timeline and decisions

| Time | Work | Decision or deliverable |
|---|---|---|
| Now–11:30 | Refresh logs every 5 minutes; check finite metrics, progress, frozen update counts, checkpoints, and evaluation output. | Eight functioning diagnostic runs; investigate errors or prolonged lack of output before interpreting learning. |
| 11:30–12:00 | Compare the last 20 training records and all four evaluation probes, separated by learner role, opponent and action mode. Compare both training seeds. | Written interpretation and a candidate recipe, or a specific failure diagnosis. Allow this window to move if evaluations are still running. |
| 12:00–14:00 | Resolve the failure branch below; run a short targeted development experiment if needed. Audit the same checkpoint's video and metrics for role consistency. | Freeze a candidate recipe by 14:00 if evidence supports one. Save exact code/config/checkpoint provenance. |
| 14:00–17:00 | If the development gate passes, start fresh confirmation seeds 2–6 plus matched control seeds 2–4 on g4's eight GPUs. Cap each recipe at 200 iterations total (409.6M steps), counting all curriculum stages. | Five candidate runs and three equal-budget controls; no recipe changes after their results are observed. |
| 17:00–18:00 | Evaluate frozen final checkpoints, inspect consistent videos, assemble separate pursuit/evasion plots, seed-level results and failures. | End-of-day evidence report. If no recipe passed development, report the blocker and next test instead of claiming confirmation. |

## Interpretation rules

The following are practical development gates, not statistical proof.
For stationary readiness, require both seeds of a condition to achieve at least
80% pursuer tags and at most 5% pursuer out-of-bounds over the final 20 records.
Also inspect trends, falls and physical survival. Mixed-league training averages
depend on sampled opponents; compare matched evaluation opponents directly.

- Stationary succeeds, mixed fails: use stationary pursuit pretraining followed
  by a declared mixed-league stage. Validate the stage transition on development
  seeds before freezing the recipe. Include both stages in the sample budget.
- Frozen clearly outperforms joint across both seeds: keep locomotion frozen
  for the next experiment. A single favorable seed does not select a recipe.
- Neither stationary condition reliably succeeds: audit role/action/relative
  observation mapping and terminal rewards, including the pursuer's zero payoff
  for leaving bounds. Examine sparse credit assignment; lambda=1.0 is a supported
  targeted test before adding a different reward objective. Fix confirmed bugs
  and test them before another broad sweep.
- Stochastic evaluation succeeds but deterministic evaluation/video fails:
  investigate the strategy action distribution and deterministic execution;
  report both protocols explicitly.
- Both stationary and mixed succeed: select the simplest condition with stable
  pursuit and survival across both seeds, then lock it for confirmation.

Report pursuer tag/success rate, evader success rate, time/steps to tag for
successful episodes, physical survival, out-of-bounds and falls. Do not select
using combined Elo or evader timeout alone. Summarize variation between training
seeds; evaluation episodes are not independent training seeds.

## Confirmation and final evaluation

Reserve independent opponent checkpoints before confirmation (for example,
baseline seed4_g4 iteration950 and the independent ablation seed2 iteration500,
after verifying availability). Their policies were not in array4199's league.
Do not tune on their outcomes: they are held out from candidate selection and
from the candidate's training league, not evidence of universal generalization.
Fix opponent identities, reset seeds, action modes and evaluation budgets in a
manifest before launching confirmation. Use at least 32 evaluation episodes per
role/opponent/action mode if runtime permits. Give the control the same total
training budget and evaluation protocol. Report failures and every training seed.

Keep the current reward objective through diagnostics. If a later fix changes
the objective, disclose it, update the baseline comparison, and distinguish the
new claim. A staged curriculum is part of the method, not an unreported warm start.

The lightweight monitor records summaries through 18:00. It does not make
training decisions or schedule future assistant replies. Follow-up launches
depend on interpreting the actual diagnostic results at the decision windows.
