# Selected completed checkpoint

The user requested consolidating on the best completed run instead of continuing
to launch ablations. The selection is **GAE lambda 1, seed 1, iteration 100**:

```text
logs/hierarchical/training/tag_pursuit_mechanism_gae100_seed1_job4324/checkpoints/000000000100
```

Verified learner-state SHA256:
`fc86798f721a0616e3422fac6ac008e8f090381bd9d751b7d01594866fb9be2c`.
The original checkpoint is unchanged. Machine-readable selection/provenance is
`logs/hierarchical/selections/tag_best_completed_20261015.json`.

## Correction: pre-fix batched contact outcomes are provisional

The pre-fix Warp detector read a globally packed contact buffer from vmapped
worlds without selecting the owning `worldid`. Consequently, a contact in one
training/evaluation world could be attributed to another world and change TAG
termination, rewards, and PPO targets. The detector and reset/state plumbing,
including explicit IDs in the single-agent P7 vector wrappers, are now fixed
and pass GPU isolation audits, but all pre-fix batched TAG rates and
contact-derived training curves remain provisional until they are regenerated.
This includes the **93/128** outside-training-panel figure below; it must not be
used as validated success evidence.

The callback's single-world videos and their associated plots are independently
role-audited: every collected sample records complementary physical-agent role
encodings, selected strategy heads, learner/opponent parameter sources, and
fixed pursuer/evader colors. That role audit does not repair the old batched
contact metrics.

## Why this one

Among the 20 completed policies covered by the matched outside-training-panel
benchmark, it has the strongest balanced pursuit/evasion result. At 2 m, across
two reference opponents, face-to-face and +/-90-degree headings, both joint
action modes, and reset seeds 5000–5015:

| Policy | Pursuer TAGs /128 | Evader tag-free full windows /128 |
|---|---:|---:|
| Selected GAE1 seed1 | 93 (72.7%) | 118 (92.2%) |
| Same-array control seed1 | 93 (72.7%) | 85 (66.4%) |
| Potential .50 seed1 | 88 (68.8%) | 70 (54.7%) |
| Small initialization seed1 | 90 (70.3%) | 40 (31.2%) |
| Global normalization seed1 | 86 (67.2%) | 43 (33.6%) |

This is retrospective model selection, not proof of statistical superiority or
an exhaustive comparison of every historical checkpoint. Other initializations
can produce a visually appealing pursuer while their evader collapses. Both-role
performance takes precedence over that demonstration. References were already
development-used, and reset episodes are not independent training seeds.

## Limits retained in the handoff

- Existing self-play gets **0/16 TAGs in each action mode** at 1.5 m. Selecting
  this policy does not make the competitive task solved.
- GAE1 seed2 does not reproduce this policy's result. This is the best observed
  model, not a demonstrated seed-robust training recipe.
- Root-distance closure, TAG completion and finite-window evasion remain separate
  diagnostics; truncation is not a terminal game win/loss.

Original W&B: https://wandb.ai/luc-ssl/MotionForge/runs/xlled6cm

Corrected role-separated plots:
https://wandb.ai/luc-ssl/MotionForge/runs/f9p6d95v

Revised dashboard (v2, identical underlying measurements):
https://wandb.ai/luc-ssl/MotionForge/runs/rtrvu0j4
The original 100 updates have 82 calibrated difficulty measurements and 18
uncalibrated batches. Grey bands mark missing measurements; calibrated-only
10-update summaries do not interpolate them. Reward, relative skill and
difficulty have separate axes for each role.

Headline overlay v3 (unbounded relative skill and difficulty rating points on the
left, actual reward on the right):
https://wandb.ai/luc-ssl/MotionForge/runs/rcavtatz
Live continuation companion: https://wandb.ai/luc-ssl/MotionForge/runs/c1yuoa4q
See `2026-10-15-dexterity-and-rating-overlay.md` for definitions and benchmark
verification. Post-continuation comparison job 4452 depends on successful 4443
completion; no extra training changes are involved.

## Exact-settings continuation to one billion total steps

Job **4443** is running on one g3 GPU, continuing iteration 100 through 489
(204,800,000 to 1,001,472,000 world steps). Complete restored learner state,
including both optimizer states, was verified against the selected payload before
the first update. Training hyperparameters, seed, opponents, rewards, architecture,
frequencies and frozen locomotion are unchanged. Only run bookkeeping, duration
and diagnostic logging differ. Source payload SHA256 remains unchanged.

Continuation W&B: https://wandb.ai/luc-ssl/MotionForge/runs/zjt88a2o

See `2026-10-15-selected-continuation-to1b.md` for launch and verification details.

## Actions after selection

- Leave all eight already-running **4415** jobs to finish, per the user's explicit
  reply. They remain comparisons, not an automatic replacement for this selection.
- **Do not submit** the prepared full previous-command ablation on g3. Its smoke
  4423 and CPU tests passed; retain the code/results for future diagnosis only.
- g3 validation **4425** freezes the selected payload hash and tests fresh reset
  seeds 8000–8063, four fixed reference checkpoints (including two additional
  motor-control seeds), 2/3 m starts, face-to-face/perturbed headings and both
  action modes. Separate fresh self-play tests and labelled successful/failing
  trajectory/video examples are included. New motor-control references are not
  pristine development-independent test policies; only this matchup panel is new.
- No further optimizer/model/reward changes are part of this consolidation.
  Report the fresh validation honestly, including failures, without retroactively
  relabeling a different checkpoint as the original frozen selection.
