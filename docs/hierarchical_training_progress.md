# Hierarchical TAG training progress

## Status

MotionForge now has an end-to-end hierarchical TAG training path that jointly
optimizes a 10 Hz strategy policy and a 50 Hz locomotion policy while executing
physics at 500 Hz. The locomotion policy is initialized from the validated P2
G1 checkpoint rather than learned from scratch. A fixed copy of that warm-start
hierarchy currently serves as the league opponent.

The current implementation can train, checkpoint, resume, evaluate, and render
the hierarchy. Two early training failures have been identified experimentally:

1. aggressive locomotion updates destroyed the P2 warm start;
2. assigning one learner role to an entire PPO batch caused role forgetting.

Both failures are fixed at the optimization/data-path level and have passing
GPU regression tests. An 80,000-step balanced-role diagnostic additionally
confirmed that both roles are retained at the early checkpoint. Its final
checkpoint exposed a separate behavioral stability problem: both roles learn
commands that eventually terminate on the arena boundary. This is not role
forgetting because every update remains exactly balanced and both roles regress
together.

## Implemented architecture

The environment hierarchy is:

```text
WarpEnv
└── LeagueEnv
    └── MultiAgentTaskEnv
```

`MultiAgentTaskEnv` owns the two-agent TAG task state, role assignment,
relative observations, tag contact detection, fall detection, arena bounds,
and timeout termination. `LeagueEnv` supplies the opponent policy. `WarpEnv`
owns the MJX-Warp physics model and stepping interface.

The learned policy has two independently optimized modules:

- Strategy policy (10 Hz): consumes opponent-relative state, arena-relative
  state, previous command, and a pursuer/evader role encoding. It emits a
  bounded planar velocity command `(vx, vy, yaw_rate)` and a value estimate.
- Locomotion policy (50 Hz): consumes proprioception, gait phase, previous
  action, and the current strategy command. It emits 29 normalized joint
  actions and a value estimate.
- Physics (500 Hz): applies each locomotion action for ten simulation substeps.

The two policies have separate PPO objectives, value heads, optimizer states,
learning rates, update schedules, gradient clipping, GAE targets, metrics, and
checkpointed counters. This keeps the hierarchy jointly trainable without
collapsing it into one monolithic action distribution.

## Training and evaluation infrastructure

The training launcher supports:

- vectorized MJX-Warp rollout collection;
- P2 locomotion warm-start migration;
- immutable versioned checkpoints;
- exact optimizer-state restoration;
- JSONL metrics and run manifests;
- W&B metric logging;
- deterministic role-specific checkpoint evaluation;
- headless MP4 rendering.

Two-agent physics uses `naconmax=256` and `njmax=512`. Earlier defaults silently
overflowed the contact buffers during training. Training, evaluation, and
rendering now expose and validate these capacities.

## Experiment 1: aggressive joint optimization

Configuration:

- 16 environments;
- 100 control steps per rollout;
- 100 PPO iterations;
- 160,000 total environment steps;
- strategy learning rate `3e-4`, updated every iteration;
- locomotion learning rate `1e-4`, updated every iteration;
- fixed P2 warm-start opponent;
- one randomly selected learner role per entire iteration.

The early checkpoint performed well, but later checkpoints catastrophically
degraded:

| Checkpoint | Pursuer success | Evader success | Evader survival |
| --- | ---: | ---: | ---: |
| iteration 10 | 100% | 100% | 100% |
| iteration 50 | 100% | 25% | 75% |
| iteration 80 | 75% | 0% | 0% |
| iteration 100 | 25% | 0% | 0% |

At iteration 100 the locomotion approximate KL reached `29.0`. Command tracking
RMSE increased from approximately `0.12–0.14` at iteration 10 to `0.65–1.02`.
This established that the final checkpoint cannot be assumed to be the best
checkpoint and that unconstrained low-level adaptation destroys the useful P2
initialization.

## Locomotion stability repair

The repaired defaults are:

- locomotion learning rate `1e-5`;
- locomotion update interval 5;
- maximum accepted pre-update locomotion approximate KL `0.01`;
- strategy learning rate and cadence unchanged;
- global gradient norm clipping remains `1.0`.

When a scheduled locomotion batch has non-finite KL or KL above the configured
limit, its parameter and optimizer-state update is rejected. Strategy learning
remains independent and proceeds normally. A regression test deliberately
constructs a stale locomotion batch and verifies that rejection leaves both the
parameters and optimizer state unchanged.

In the matched 160,000-step stability experiment, locomotion received 20 updates
while strategy received 100. Final locomotion KL was `3.5e-7`, all metrics were
finite, and no KL rejection was required. Thus the low-level numerical failure
was removed.

## Experiment 2: stabilized but single-role batches

The stabilized controller retained low command-tracking error, but role-level
performance still oscillated:

| Checkpoint | Pursuer success | Evader success | Tracking RMSE range |
| --- | ---: | ---: | ---: |
| iteration 10 | 100% | 100% | 0.11–0.12 |
| iteration 50 | 0% | 0% | 0.16–0.28 |
| iteration 100 | 0% | 100% | about 0.28 |

Unlike the first experiment, this was not locomotion collapse. The training
launcher chose one learner role for all 16 environments in an iteration. Long
sequences of role-specific updates moved the shared strategy toward one role and
away from the other. The role-conditioned observation was present, but the batch
distribution did not force each gradient estimate to represent both roles.

## Balanced-role batch repair

Every batched PPO iteration now contains an equal number of pursuer and evader
environments. Both role groups:

- use the same current learner parameters;
- independently execute against the frozen league opponent;
- compute bootstrap values with the correct learner index;
- compute role-specific GAE targets;
- merge into one flattened strategy batch and one flattened locomotion batch;
- contribute together to one optimizer update.

MJX-Warp `Data` objects cannot be sliced or merged with a generic tree operation
because their contact arrays have special capacity axes. The implementation
therefore collects both role alternatives over the intact vectorized state and
uses `mjx.Data.where` to select the appropriate per-environment physics result.
Ordinary trajectory arrays are selected with broadcast masks.

The GPU regression test verifies:

- exactly 50% pursuer and 50% evader strategy samples in one update;
- all vectorized environments advance;
- environment identifiers are preserved;
- stochastic rollouts remain independent;
- strategy and locomotion batches have their expected flattened shapes;
- all metrics remain finite;
- both parameter groups update on their scheduled step.

## Experiment 3: balanced-role diagnostic

Configuration:

- 16 environments, split evenly between pursuer and evader on every update;
- 100 control steps per rollout;
- 50 PPO iterations;
- 80,000 total environment steps;
- repaired locomotion learning rate, update interval, and KL guard;
- fixed P2 warm-start opponent.

The training run completed with finite metrics, ten accepted locomotion updates,
50 strategy updates, and no rejected locomotion update. Every iteration reported
a pursuer sample fraction of exactly `0.5`. Final locomotion approximate KL was
`4.1e-7`, so the low-level warm start remained numerically stable.

| Checkpoint | Pursuer success | Evader success | Physical survival | Tracking RMSE (P/E) |
| --- | ---: | ---: | ---: | ---: |
| iteration 10 | 100% | 100% | 100% / 100% | 0.113 / 0.119 |
| iteration 50 | 0% | 0% | 0% / 0% | 0.182 / 0.146 |

At iteration 10 the pursuer tagged in all four seeded rollouts and the evader
avoided tagging in all four. At iteration 50 neither role failed because it had
been omitted from training: both were present in every update. Instead, the
learned command norms increased and all eight evaluation rollouts eventually
failed the physical-survival criterion after moving toward or through the arena
boundary. Mean final distance grew to `3.18 m` for the pursuer and `4.48 m` for
the evader.

This experiment validates the role-forgetting repair while showing that balanced
sampling alone does not guarantee monotonically improving behavior. For the
current fixed-opponent curriculum, iteration 10 is the best validated checkpoint;
the final checkpoint should not be selected merely because it is latest.

## Interpretation

These experiments support retaining a hierarchical design while controlling the
optimization timescales. The P2 controller is a useful initialization, not an
immutable subsystem: it may adapt to difficult TAG states, but its updates must
be slower and trust-region guarded. Simultaneously, role conditioning alone is
not sufficient to prevent interference. Each optimization batch must represent
both roles.

The staged diagnostics also demonstrate why checkpoint evaluation is part of
training rather than a final afterthought. Aggregate finite-loss checks did not
detect either behavioral regression. Role-specific success, survival, command
tracking, root height, and tilt measurements did.

## Next experiments

1. Add arena-aware command regularization or curriculum constraints so strategy
   improvement cannot be obtained by driving indefinitely toward a boundary.
2. Evaluate intermediate balanced checkpoints to locate the onset of boundary
   termination, then rerun the 160,000-step experiment after that repair.
3. Select checkpoints using a conservative score such as the minimum of pursuer
   and evader success, with survival and tracking constraints.
4. Render both roles from the selected checkpoint.
5. Once fixed-opponent training is stable, introduce historical league snapshots
   and then controlled self-play mixtures.
6. Use TAG failure trajectories to identify difficult locomotion states for
   targeted low-level training and ablation studies.

The current evidence does not yet justify claiming a converged TAG policy. It
does justify that the hierarchical execution, optimization, checkpointing,
evaluation, contact capacity, locomotion stability, and balanced-role data path
are implemented and individually validated.
