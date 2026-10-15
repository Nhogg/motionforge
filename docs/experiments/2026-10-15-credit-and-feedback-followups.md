# Follow-ups after completed motor-noise and transfer studies

## Decision from completed results

See `2026-10-15-motor-noise-and-transfer-results.md` for exact per-seed tables.
All eight motor runs finished all three evaluation modes. Disabling motor
sampling during training made seeds 0 and 2 substantially worse and did not
consistently improve the others. Do not promote that recipe. The switch also
affected opponent motors, so it was not a learner-only actuator intervention.

The complete 12-run candidate/control transfer panel shows no reliable winner.
GAE lambda 1 has the most balanced individual exemplar (seed 1), but seed 2
trades pursuit against evasion. Stronger shaping, small initialization and
normalization changes do not survive all seed/role/opponent checks. We therefore
keep potential alpha .25 and the existing motor sampling as the common baseline.

## g4: credit assignment and longer optimization

GPU smoke **4414** passed with finite metrics and checkpoint completion. Array
**4415** uses eight GPUs, four paired seeds 0–3:

- Control: GAE lambda .95.
- Test: GAE lambda 1.0.

Both train fresh strategy for **500 iterations**, rather than stopping after
100. This is 1.024B world steps and 500 strategy updates. The existing 100-update
results are variable; this study asks whether the apparent credit-assignment
benefit reproduces and persists with longer optimization. It does not assume
that more training solves the task. Checkpoint 100 is evaluated separately from
500, against identical reference conditions; never label 100 as the final model.

Other settings are unchanged: frozen accepted P2 locomotion; strategy/motor
sampling enabled; alpha .25, smoothness .005; same .75 historical mixture,
1.5 m initial separation, 10-second truncation, architecture, optimizer and
execution frequencies. 12-hour Slurm limit accommodates training plus evaluation;
this is a longer-budget experiment, not a free speed improvement.

W&B group `tag_credit_horizon`; launcher
`scripts/slurm/tag_credit_horizon_g4.sbatch`.

## g3: previous-command feedback mechanism

Prior CPU actor probes found learned command bias amplified by previous-command
feedback with weak bearing sensitivity. This is a hypothesis, not a proven
cause. The second paired study tests whether learning can avoid this shortcut.

Four seeds 0–3, control versus learner previous-command connections held at zero,
**200 iterations** / 409.6M world steps per run. GAE stays .95; all other common
settings above remain unchanged. No combination with lambda 1 or motor-noise
changes in this first test.

The network retains all 11 input dimensions. Only rows 6:9 of the learner's first
strategy encoder kernel are initialized and constrained to zero. Projecting the
strategy loss gives exactly zero gradients in those rows; fresh Adam moments
there remain zero. Motor weights, role heads, other encoder inputs, reward
computation and historical-opponent weights are untouched. This also removes
previous-command information from the shared strategy value representation;
it is not an actor-only test. Previous-command information may be useful in
partially observed control, so removing it is not presumed beneficial.

A global observation mask was deliberately avoided: that would also change
the historical opponents' input protocol. Here zero-row checkpoints enforce
the learner constraint through ordinary inference, with no special renderer or
evaluation mask. Fresh-only masked training avoids actor/loss disagreement or
nonzero restored optimizer moments. Control delegates the original updater.

Launcher `scripts/slurm/tag_command_feedback_g3.sbatch`, W&B group
`tag_command_feedback`. CPU invariance/gradient/Adam/serialization checks and
a GPU checkpoint-reload smoke are required before dispatching the full array.

## Matched evaluation and plots

`tag_followup_panel.sh` evaluates checkpoints 0/100/final separately. Each gets
the same two development historical references, distances 1.5/2 m, both action
modes and 16 reset seeds per role; self-play is separate. Final checkpoints also
get the same two outside-panel references, face-to-face and +/-90-degree headings,
2 m, both action modes, and two role-separated real trajectory plots. Those
references remain development-used, not a pristine test set.

Both new studies enable live calibrated role-separated skill/difficulty logging
using the completed reference collection. All 32 earlier companion plots were
published; original raw histories are untouched. Relative skill is not empirical
TAG probability, and the weak evader anchor limits interpretation. Independent
checkpoint evaluation takes precedence over attractive training curves.

Promotion requires a repeatable pursuit improvement across multiple training
seeds, without evader collapse or increased own fall/OOB, and evidence beyond
face-to-face training-league opponents. No selection from a single video, one
seed, self-play TAGs, or mean reward. These are exploratory paired studies, not
preregistered statistical-superiority claims. If neither works, report that
negative evidence and inspect actual directional commands before scaling again.
