# Joint locomotion training and renderer role audit

## What the joint/frozen experiments establish

Array 4199 used fresh strategies with pretrained P2 locomotion. In mixed-league
seed 0, the last-20 training pursuer out-of-bounds rate was 86.15% with joint
training and 8.03% frozen. Deterministic evaluation physical survival was
37.5% joint versus 100% frozen, but both had zero tags. Seed 1 tagged in every
tested development episode with either update scheme. Full results are in
2026-10-14-tag-diagnostics-results.md.

These observations support freezing as an isolation/stability measure. They do
not show that locomotion updating caused all pursuit failures. The strategy
still fails in seed 0 with locomotion frozen.

Plausible mechanisms to test: changing low-level command response creates a
moving target for strategy learning; movement reward and sparse task payoff
optimize different things; both effects can interact with seed-sensitive
strategy updates. These are hypotheses, not experimentally isolated causes.
Future joint tuning should be compared against a reliable frozen baseline with
matched seeds, command tracking, physical survival and separate role outcomes.

## Renderer roles and head selection

The code path is:

1. Runner reset calls role_encodings(environment.config.pursuer_index).
2. Default pursuer_index=0 produces agent0=[1,0] and agent1=[0,1].
3. Strategy observations append each physical agent's own encoding.
4. StrategyModule selects pursuer_head for [1,0], evader_head for [0,1].
5. The learner role chooses which physical agent gets the learner parameter
   tree. It does not change both agents' role encodings.

The CPU regression test test_hierarchical_renderer_roles.py forces the two
heads to opposite commands and gives learner/opponent different magnitudes.
It verifies both physical pursuer indices and both learner roles. An invalid
two-evader encoding is rejected. The existing role-head gradient/migration
test also passes. Actual rollout diagnostics validate and record these roles.

Older shared-head checkpoints migrate their head parameters into both new
head slots. That migration does not assign both agents the evader role:
the encoder still receives separate role encodings. The trajectory JSON
records whether legacy-head migration occurred.

## A real presentation ambiguity

Previously green meant LEARNER and orange meant OPPONENT. Thus green was a
pursuer in a pursuer video and an evader in an evader video. Reading green as
"always pursuer" reverses the interpretation of the evader video.

New videos always use GREEN=PURSUER and ORANGE=EVADER, print those labels in
the video, and save physical-agent roles, selected heads and learner/opponent
indices in metadata. Existing MP4s are unchanged and retain their old colors.
Already-running training processes retain the callback implementation loaded
at process startup.

Visual QA also found that the old code recolored only contact-layout collision
geoms, leaving visible mesh geoms/materials white. New code resolves every
agent geom through its body's agent prefix and removes material overrides in
the native rendering model so the role colors are actually visible. This
does not change the already-compiled Warp physics model.

There is another matchup distinction: the training callback uses CURRENT
SELF-PLAY, putting the checkpoint parameters on both agents with different
role encodings. It inherits stochastic action sampling from the training
runner. The standalone renderer defaults to a WARM-START opponent with
deterministic actions. These are different experiments. The renderer now has
explicit use_self_play, opponent_checkpoint, stochastic_actions and
use_training_reset options so the matchup can be reproduced deliberately.

Switching the selected learner role during self-play with identical policies
does not change the underlying physical roles. The two old videos highlighted
different agents in the same underlying self-play matchup.

## Reproduce a checkpoint trajectory

The plotter produces PNG, SVG and raw JSON with world-space paths, start/end
markers, direction arrows, separation versus time, and radial movement versus
the opponent. Negative pursuer radial speed means the pursuer itself moves
away; increasing separation alone can also mean the evader is faster.

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/evaluation/plot_hierarchical_tag_trajectory.py \
  checkpoint=logs/hierarchical/training/tag_clean_start_mixed_4096_seed3_g4/checkpoints/000000001250 \
  use_self_play=true use_training_reset=true stochastic_actions=true seed=3 \
  output=logs/hierarchical/trajectories/seed3_iter1250.png
```

Run simulation commands on an allocated GPU. Matching the training-video seed,
action sampling, reset range and matchup is necessary for a useful comparison.
An explicit initial_separation override fixes the spawn distance. Supplying
opponent_checkpoint selects a particular opposing policy. The plotter defaults
to self-play; the standalone video renderer preserves its warm-start default.

For a newly labeled video of the same matchup, use render_hierarchical_tag.py
with use_self_play=true use_training_reset=true stochastic_actions=true seed=3,
the same checkpoint, and output=...mp4.

Eight GPU audit rollouts compare the late seed-3 checkpoint at iteration1250
(2.56B environment steps), the successful mixed-frozen seed-1 checkpoint,
deterministic/stochastic execution, and both learner highlights. Their actual
plots and numeric conclusions are recorded after completion.

## GPU audit results

The real checkpoint rollouts recorded physical roles [[1,0],[0,1]], selecting
pursuer_head and evader_head respectively. No duplicated evader encoding was
found. The late checkpoint already has separate heads; it did not require
legacy-head migration. The GPU callback test confirms fixed physical roles,
correct learner index for both video names, and fixed color semantics.

| Checkpoint and matchup | Action mode | Result | Mean pursuer velocity toward evader |
|---|---|---|---:|
| Seed3 iteration1250 (2.56B), self-play | Deterministic | No tag; pursuer OOB at 5.52 s | -0.281 m/s |
| Same late checkpoint, self-play | Stochastic | No tag; pursuer OOB at 6.40 s | -0.384 m/s |
| Mixed-frozen seed1 iteration100, historical005 | Deterministic | Tag at 2.08 s; no falls/OOB | +0.727 m/s |
| Same seed1 checkpoint, historical005 | Stochastic | Tag at 2.00 s; no falls/OOB | +0.724 m/s |
| Same seed1 checkpoint, self-play | Deterministic | No tag; pursuer OOB at 8.84 s | -0.044 m/s |
| Same seed1 checkpoint, self-play | Stochastic | No tag; pursuer OOB at 8.04 s | -0.080 m/s |

For the late stochastic example, pursuer radial movement was below -0.05 m/s
for 96.56% of recorded intervals. Initial separation was 2.727 m; it ended at
4.653 m. This directly supports genuine retreat, independent of camera or
video labels. These numbers describe selected single episodes, not a new
multi-seed benchmark.

Changing the learner highlight from pursuer to evader preserved both agents'
role encodings, initial world positions and first executed commands exactly.
Later rollouts on separate GPUs showed small numerical differences, so the
audit does not require complete bitwise trajectory equality.

Important interpretation update: the seed1 checkpoint's 100% development
historical-opponent evaluation result does not establish successful self-play.
The self-play path initially chases, then splits away as separation grows.
Future evaluation must include self-play and stronger opponents alongside the
older historical policies; renderer labeling is not the pursuit-learning fix.

Artifacts:

- Late failure and self-play comparisons: logs/hierarchical/trajectories/role_audit_job4257/
- Historical-opponent tagging comparison: logs/hierarchical/trajectories/role_audit_job4265/
- Newly labeled, visible-color video: logs/hierarchical/trajectories/role_render_job4268/
- Raw trace JSON and editable SVG accompany every PNG.
