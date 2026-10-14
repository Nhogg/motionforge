# Array 4199: pursuit diagnostic results

All eight runs completed 100 iterations / 204.8M environment steps. All 32
evaluation files passed execution checks. The evaluation matrix contains 1,024
episodes: eight training runs x two opponents x two action modes x two roles x
16 reset seeds. Those episodes are repeated tests of only two training seeds.

## Main finding

Mixed-league training produced successful pursuit for training seed 1 under
both joint and frozen locomotion. Training seed 0 failed pursuit under every
condition. There is a successful candidate checkpoint, but no condition meets
the planned readiness gate across both training seeds.

Subsequent GPU trajectory audits also found that this seed1 checkpoint fails
in current-policy self-play despite its historical-opponent success. See
hierarchical-tag-locomotion-and-renderer-audit.md for the role-routing audit,
world-space traces and revised interpretation. Its success is matchup-specific.

## Final checkpoint evaluation

Counts pool historical opponents 005 and 020: 32 pursuer episodes per action
mode. They are development opponents, including training opponents for mixed
runs. Evaluation uses the default 2 m separation; training used fixed 1.5 m.

| Condition | Training seed | Deterministic pursuer tags | Stochastic pursuer tags | Pursuer physical survival, deterministic / stochastic |
|---|---:|---:|---:|---:|
| Mixed, joint | 0 | 0/32 | 0/32 | 37.5% / 31.25% |
| Mixed, joint | 1 | 32/32 | 32/32 | 100% / 100% |
| Mixed, frozen | 0 | 0/32 | 0/32 | 100% / 100% |
| Mixed, frozen | 1 | 32/32 | 32/32 | 100% / 100% |
| Stationary, joint | 0 | 0/32 | 0/32 | 100% / 100% |
| Stationary, joint | 1 | 0/32 | 7/32 | 100% / 100% |
| Stationary, frozen | 0 | 0/32 | 0/32 | 100% / 96.875% |
| Stationary, frozen | 1 | 0/32 | 10/32 | 100% / 87.5% |

For mixed seed 1, mean successful time to tag was 1.792 seconds deterministic
and 1.877 seconds stochastic for joint training; frozen training gave 1.790
and 1.876 seconds. Both learner roles achieved 100% role success in these
development evaluations for this seed. Physical survival here means no
learner fall or out-of-bounds event, not full-episode survival after a tag.

Seed 0's mixed-joint evader success was 78.125% deterministic / 87.5%
stochastic. Mixed-frozen evader success was 84.375% / 53.125%.
Freezing therefore did not consistently improve both roles.

## What changed during training

Pursuer tag rates averaged over the first and last 20 training records:

| Condition | Seed 0: first -> last | Seed 1: first -> last |
|---|---:|---:|
| Mixed, joint | 15% -> 0% | 50% -> 100% |
| Mixed, frozen | 15% -> 0% | 50% -> 100% |
| Stationary, joint | 25% -> 0% | 50% -> 100% |
| Stationary, frozen | 30% -> 0% | 50% -> 100% |

These are training-batch summaries, not matched held-out trials. Mixed-league
opponent selection varies between records. Nevertheless, seed 0 had successful
tag batches early, so its final failure is not simply that tags were never
encountered. It lost pursuit while seed 1 improved.

Seed 0's last-20 pursuer out-of-bounds rate was 86.15% for mixed-joint and
8.03% for mixed-frozen. Frozen runs recorded zero locomotion updates.
Freezing helps physical stability in that comparison, but leaves tags at zero.
The strategy training path remains a leading concern; these results do not
identify an exact implementation bug or prove a reward change is necessary.

Stationary seed 1 achieved 100% training tags at 1.5 m yet zero deterministic
tags against the historical evaluation opponents at 2 m. Stochastic tags were
21.875% (joint) / 31.25% (frozen). Stationary mastery alone did not transfer to
the changed opponent and distance.

## Decision

- Keep mixed-frozen seed 1 as a provisional working checkpoint for videos and
  additional evaluation; joint seed 1 performs essentially identically here.
  This is a safety-oriented provisional choice, not evidence of superior tag
  performance across training seeds.
- Evaluate it against the two reserved independent opponents before claiming
  transfer. These existing 005/020 probes are not held-out tests.
- Expand the development seed count and investigate why seed 0 loses pursuit.
  Prioritize strategy return/advantage/command evolution and credit assignment;
  freezing locomotion did not resolve the failure. Do not launch a long final
  confirmation sweep on the strength of one favorable training seed.
- Match deterministic and stochastic protocols explicitly. The mixed seed-1
  result holds in both, so its success is not solely an action-noise effect.

The evaluator randomizes global spawn rotation but retains symmetric,
face-to-face starts. Broader starting configurations and held-out opponents
are needed for broader claims. A true untrained strategy evaluation was not
part of this matrix; add it when measuring learning gain over initialization.

## Sources

- Training: logs/hierarchical/training/tag_pursuit_diag_*_job4199/metrics.jsonl
- Completion: each run's summary.json
- Evaluation: each run's evaluation/vs_*_stochastic_*.json
- Live generated summary: logs/hierarchical/diagnostics/job4199/status.md
