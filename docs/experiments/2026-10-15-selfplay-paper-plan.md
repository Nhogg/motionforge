# Plan: demonstrate repeatable learned-policy self-play

**Execution update:** recovery 4473 and dependent report 4481 were stopped after
stalled current-policy TAG metrics. Checkpoint evaluation 4486 and short,
imitation-assisted/curriculum pilots 4492 replace the remaining budget. See
`2026-10-15-pursuit-rescue.md` for the diagnosed direction-independent command
bias, explicit initialization changes, smoke gates and preserved experiments.

> **Contact-label correction:** Pre-fix batched Warp contact outcomes are
> provisional. The globally packed contact buffer was not isolated by
> `worldid`, so any pre-fix batched TAG rate, terminal outcome, reward curve,
> or PPO target must be regenerated before it is used as evidence. A GPU
> two-world isolation regression now passes; this plan's self-play panels
> remain pending corrected reruns.

## Required outcome

Self-play is a required empirical result for the paper, not an optional diagnostic.
The goal is genuine learned pursuer-versus-learned evader behavior, valid TAG
contacts, stable locomotion, and evidence of improvement under competitive
training. Existing failed development results remain in the experiment record;
they cannot be relabeled as successes. Publication readiness requires passing
the gates below; it is not guaranteed by elapsed training or attractive plots.

## Current evidence and causal hypotheses

The selected checkpoint's existing 16-seed self-play tests start at 1.5 m.
Deterministic mean minimum separation is 1.28 m and final separation about 6.97 m;
stochastic mean minimum is 1.19 m and final about 5.90 m. Neither reaches .5 m,
and physical survival is 100%. For these episodes, the leading problem is
failure to approach, not failure to register contact after reaching the opponent.

The old mixture provides only 8.33% current-policy self-play, against 75%
historical opponents. Those historical matchups often have 95–100% training
TAG rates while current-policy batches have none. This makes historical
specialization a plausible, but unproven, cause. Earlier 50%-recent experiments
used different checkpoints/reward/GAE conditions and did not establish a fix.
The new test is not claimed to be the first exposure experiment.

Previous CPU strategy probes also found previous-command feedback dominating
bearing sensitivity. A tested learner-only zero-row constraint is available;
it has not yet been shown to improve actual self-play.

## Phase 1: targeted parallel diagnosis — now

GPU array **4467**, g4, eight tasks/max three concurrent, does read-only tests:

1. Selected learned policy versus itself, deterministic/stochastic, 10 s.
2. The same self-play at 30 s, explicitly a longer observation window, not a
   substitute for the 10 s result or a new timeout game payoff.
3. Selected policy against scripted pursuit/evasion, separately labelled.
   The learner-as-evader scenario tests whether a responsive pursuer can catch
   the selected evader; the learner-as-pursuer scenario tests active pursuit
   against a moving scripted evader. These are NOT self-play evidence.
4. New joint-motor checkpoint 50, true learned self-play, both action modes.

All use 16 development reset seeds 23000–23015, 1.5 m starts, original arena,
TAG detector, payoff and command limits. Save raw trajectories, minimum/final
separation, terminal causes and valid TAGs. Do not change speed limits, tag
radius or payoff to force a successful outcome.

First completed diagnostic results: selected-policy 10 s self-play has 0/16
valid TAGs in each action mode. Deterministic 30 s self-play also has 0/16
TAGs, with 12/16 evader OOB endings in the canonical pursuer scenario. Simply
extending the window is therefore not supported as the fix. Against the
scripted evader, the learned pursuer tags in 4/16 deterministic trials; the
learned evader avoids the scripted pursuer in all 16. These scripted matchups
are diagnostic only, not evidence that learned-policy self-play works.

## Phase 2: paired recovery study — now

Three real-GPU smoke tasks in **4466** passed training/checkpoint audits.
Production array **4473** uses six g3 GPUs, two paired seeds per recipe:

| Recipe | Historical | Current learned self-play | Stationary | Scripted |
|---|---:|---:|---:|---:|
| Matched control | 75% | 8.33% | 8.33% | 8.33% |
| Self-play emphasis | 15% | 75% | 5% | 5% |
| Self-play emphasis + feedback constraint | 15% | 75% | 5% | 5% |

Each starts a fresh optimization stage from the same selected strategy/motor
checkpoint, not its old optimizer moments. Motors stay frozen to isolate strategy
learning. 1024 worlds, 500 control steps, 10 s, 100 iterations (51.2M new world
steps), GAE lambda 1, strategy LR 3e-4, same potential alpha .25 and game rules.
The feedback variant additionally zeros learner encoder input rows 6:9 and keeps
their gradients/Adam moments zero, with the original 11-input architecture.
This is explicitly an initialization/feature-constraint intervention. Frozen
historical policies are not altered. Ordinary checkpoint inference enforces
the constraint; no special evaluator observation mask is required.

Checkpoints 0/50/100 receive matched true-self-play tests with development seeds
24000–24015, both action modes. Endpoint evaluation also includes the common
fixed-opponent panel and real trajectories. Control and treatments are all
reported. A copied selected strategy plus further training is not called a
from-initialization training result.

Evaluator metadata now records learner/opponent checkpoint SHA256 fingerprints.
True-self-play mode rejects scripted controller overrides. The report requires
identical fingerprints. Since both learner-role scenarios replay the same
policy/reset matchups, canonical pursuer episodes are counted once, never as 2N.

CPU report job **4481** runs after successful completion of **4473** and writes
`logs/hierarchical/selfplay_recovery_reports/job4481/report.{json,md}`.

### Development promotion gate

A recipe may advance only if BOTH development training seeds achieve:

- At least 25% valid TAGs at 10 s in EACH deterministic/stochastic self-play test.
- At most 10% physical-ending failures in those small development samples.
- At least 75% tag-free physically survived windows as evader across the fixed
  128-episode cross-play panel, to reject a solution that merely ruins evasion.
- Passed checkpoint, frozen-motor and variant-constraint audits.

These are small development gates, not paper-level statistical evidence. Record
initial versus final behavior; a favorable initialization alone is not learning.
No variant is presumed to pass. The report does not manufacture promotion when
none passes.

## Phase 3: confirmation after a recipe passes

Use the winning recipe for three independent fresh strategy seeds, initialized
with the disclosed accepted P2 motor (not the selected learned strategy). Keep
the best recovery continuation as separate tuning/demo evidence. Train and
evaluate checkpoints across the learning curve, with unchanged game parameters.
Do not pick one training seed and call it robust self-play.

Freeze the confirmation protocol BEFORE reading confirmation outcomes:

- 100 fresh reset/noise seeds per condition, not 23000/24000 development seeds.
- Both deterministic and stochastic action modes.
- Nominal 1.5 m and transfer 2 m starts; face-to-face and perturbed headings.
- Success by 10 s, separately success by 30 s, and time-to-valid-TAG distributions.
- TAGs, physical failure causes and truncations separately; no reward-as-win.
- Learner-versus-itself, cross-seed learned matchups and fixed reference opponents.
- At least 25% nominal 10 s TAG rate across all three confirmation seeds and both
  modes; <=5% physical-ending failures; cross-play evasion >=75%. Transfer
  conditions must be reported even if weaker; no hidden protocol relaxation.
- Report binomial uncertainty per condition and training seeds separately.
  Thresholds are engineering promotion criteria, not a guarantee of acceptance.

Include predefined-seed trajectories/videos and summaries over all episodes.
Successful-contact examples should be real episodes, accompanied by the measured
population rate. Do not present a scripted agent or a weaker checkpoint as
current-policy self-play.

## Fallback decisions if recovery gates fail

- If scripted pursuit can tag the selected evader but learned pursuit cannot:
  prioritize reactive pursuit/interception learning. A separate, disclosed
  pursuer demonstration/distillation warm start followed by actual self-play is
  a possible next experiment; it is not silently substituted for pure RL.
- If both learned and scripted pursuit fail to approach: inspect command-frame
  tracking, yaw/translation coupling and interception geometry from saved traces.
- If trajectories reach contact distance without valid TAGs: audit contact geoms
  and timing with controlled fixtures before proposing separate contact shaping.
- If longer windows work but 10 s does not: report time-to-tag and window
  dependence explicitly. Do not change the cutoff while retaining a 10 s claim.
- Do not launch another large blind sweep or alter game rules without a diagnosed
  mechanism. An inability to pass the gates is a reason for more development,
  not authorization to conceal a material result.

## Preserve the other evidence streams

Keep **4443** (selected checkpoint to 1B) and dependent comparison **4452**.
Keep **4454** (non-TAG baseline evaluation), joint motor-training **4459_3–5**,
and dependent motor comparison **4465**. The three command-drill arms have
already completed and their parameter/update audits passed.

Motor dexterity remains an independent claim requiring matched held-out physical
evaluation, not training reward or self-play TAG counts. Self-play recovery
results do not retroactively unfreeze 4443's motor weights. If motor-learning
benefits are demonstrated, show them as a separate result/ablation.

## Timing and deliverables

Run diagnosis/recovery alongside the preserved work, using the six free g3 GPUs
and at most three spare g4 GPUs. Expect the next decision after training plus
evaluation, roughly hours rather than minutes; measured job progress overrides
estimates. Confirmation may require additional hours and is conditional on
actual evidence, not an unconditional end-of-day promise.

Required package: reproducible configs/checkpoint hashes, self-play learning
curves, canonical-episode tables with uncertainty, cross-play matrix, physical
failure breakdown, predefined-seed videos/paths, and independent motor comparison.
The paper is ready to claim self-play only when the real tests demonstrate it.
