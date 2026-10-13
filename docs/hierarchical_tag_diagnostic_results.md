# Hierarchical TAG diagnostic results

## Purpose

This diagnostic pass was designed to answer four questions before any additional
training:

1. Does stochastic action sampling recover behavior that is absent from the
   deterministic policy mean?
2. Does evaluation against the real historical league agree with evaluation
   against the previous randomly initialized warm-start opponent?
3. Did the 1.16-million-step historical continuation improve cross-play against
   the accepted iteration-20 checkpoint?
4. Are pursuit failures caused primarily by locomotion failure, command tracking,
   or the high-level strategy?

All reported rates use 16 held-out reset seeds, 20--35, with 10-second episodes.
Both learner roles are evaluated independently. The matrix contains 16 evaluation
conditions and 512 total role rollouts.

## Evaluation correction

The old evaluator always installed `warm_start_parameters` as the opponent. Its
locomotion parameters came from P2, but its strategy parameters were randomly
initialized from the evaluation seed. Consequently, it was not a historical
cross-play evaluation.

The evaluator now accepts `opponent_checkpoint`. When supplied, the complete
frozen hierarchical policy is restored and used as the opponent. The report
records the action mode, opponent provenance, 10 Hz distance traces, planar paths,
strategy commands, and command-tracking errors. The old warm-start mode remains
available as an explicit baseline.

## Principal results

| Learner | Opponent | Action mode | Pursuer success | Evader success | Physical survival |
|---|---|---:|---:|---:|---:|
| iteration 10 | iteration 20 | deterministic | 37.5% | 100% | 100% |
| iteration 10 | iteration 20 | stochastic | 62.5% | 43.75% | 100% |
| iteration 20 | iteration 10 | deterministic | 0% | 50% | 100% |
| iteration 20 | iteration 10 | stochastic | 50% | 37.5% | 100% |
| iteration 20 | warm-start strategy | deterministic | 0% | 100% | 100% |
| iteration 20 | warm-start strategy | stochastic | 25% | 100% | 100% |
| iteration 145, seed 0 | iteration 20 | deterministic | 0% | 100% | 100% |
| iteration 145, seed 1 | iteration 20 | deterministic | 0% | 100% | 68.75% pursuer, 100% evader |
| iteration 145, seed 2 | iteration 20 | deterministic | 0% | 100% | 100% |
| iteration 145, seed 3 | iteration 20 | deterministic | 0% | 100% | 50% pursuer, 100% evader |

The reciprocal cross-play rows are consistent: iteration 10 tags iteration 20 in
six of sixteen deterministic episodes when evaluated directly, while iteration 20
reports eight of sixteen evader failures when iteration 10 is the frozen pursuer.
The small difference is expected because the learner and opponent occupy opposite
physical agent slots in the reciprocal tests.

## Interpretation

- A deployable deterministic pursuit policy existed at iteration 10 and degraded
  by iteration 20. Pursuit was therefore learned and subsequently forgotten; it
  was not prevented by the robot model or locomotion controller.
- Stochastic sampling exposes substantially more pursuit capability than the
  policy mean. Iteration 20 improves from 0% to 50% against iteration 10 when both
  hierarchies sample actions. The current PPO objective can therefore retain useful
  probability mass without producing a useful deterministic mean.
- The accepted iteration-20 checkpoint was selected from too narrow a seed set.
  It had appeared balanced on seeds 4--19, but its pursuer success is 0% on seeds
  20--35.
- All four long historical continuations retain 0% deterministic pursuer success.
  They do not repair the policy mean. Two seeds also damage physical survival, so
  continued joint optimization is beginning to erode locomotion.
- Locomotion is not the primary iteration-20 failure. Deterministic iteration-20
  command-tracking RMSE remains approximately 0.10--0.12 with 100% survival. The
  trajectory plots show valid locomotion while the separation ultimately grows.
  The incorrect behavior originates in strategy commands and role retention.
- Stochastic locomotion raises tracking RMSE to approximately 0.43--0.44. It can
  improve tag rate, but it is unsuitable as the deployment solution and obscures
  whether strategy or motor exploration caused a success.

## Required repair before another long run

1. Use iteration 10, not iteration 20 or iteration 145, as the behavioral control
   checkpoint for pursuer comparisons.
2. Make deterministic held-out cross-play a checkpoint gate during training. A
   checkpoint must not be promoted unless both roles retain their required success
   and physical-survival floors.
3. Separate strategy and locomotion stochasticity in rollout configuration and
   evaluation. This will identify whether strategy sampling alone explains the
   recovered tags.
4. Reduce or anneal strategy entropy and constrain strategy log standard deviation
   late in training. The final policy mean, not merely sampled actions, must solve
   pursuit.
5. Add successful-rollout mean distillation: use successful stochastic strategy
   actions as supervised targets for the deterministic strategy mean. This directly
   converts discovered pursuit behavior into deployable behavior.
6. Protect locomotion with a held-out command-tracking and survival gate. Reject or
   roll back locomotion updates that regress either metric. The seed-1 and seed-3
   continuation failures show that the present KL guard alone is insufficient.
7. If shared-head training continues to forget one role, retain the shared strategy
   trunk but introduce role-specific command and value heads. This preserves shared
   perception while preventing pursuer and evader gradients from overwriting the
   same final mapping.

## Results available for presentation

The following artifacts are already generated:

- `logs/hierarchical/plots/diagnostics_overnight/summary.svg`: combined success,
  survival, and tracking comparison.
- `logs/hierarchical/plots/diagnostics_overnight/summary.csv`: paper/table-ready
  aggregate values.
- `logs/hierarchical/plots/diagnostics_overnight/*_trajectories.svg`: per-condition
  distance and planar trajectory plots.
- `logs/hierarchical/evaluation/diagnostics_overnight/*.json`: complete metrics and
  per-seed trajectories.

These baseline results are sufficient for a progress presentation: they establish
the evaluation correction, demonstrate deterministic role forgetting, show that
stochastic exploration retains latent pursuit, and rule out basic locomotion
failure as the primary cause.

## Reproduction

Run the full resumable matrix:

```bash
.venv/bin/python scripts/evaluation/run_hierarchical_tag_diagnostics.py
```

Regenerate plots without rerunning physics:

```bash
.venv/bin/python scripts/evaluation/plot_hierarchical_tag.py \
  evaluation_directory=logs/hierarchical/evaluation/diagnostics_overnight \
  'pattern=*.json' \
  'trajectory_pattern=*.json' \
  output_directory=logs/hierarchical/plots/diagnostics_overnight
```
