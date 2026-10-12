# Hierarchical TAG training progress

## Status

MotionForge now has an end-to-end hierarchical TAG training path that jointly
optimizes a 10 Hz strategy policy and a 50 Hz locomotion policy while executing
physics at 500 Hz. The locomotion policy is initialized from the validated P2
G1 checkpoint rather than learned from scratch. Training can use either a fixed
warm-start opponent or an immutable population of historical hierarchical
checkpoints as league opponents.

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

## Component-level model description

- **Physics and environment base (`WarpEnv`).** The physical substrate is a
  two-G1 MuJoCo model executed through MJX-Warp on the GPU. Physics advances at
  500 Hz using a 0.002-second integration timestep. The environment stores the
  combined generalized position, velocity, actuator, contact, and sensor state
  for both 29-actuator humanoids. One 50 Hz motor decision is held for ten
  physics substeps. This layer is deliberately task-agnostic: it owns model
  construction, reset state, batched physics stepping, and the partition that
  maps each agent's 29 targets into the combined 58-actuator model.

- **League layer (`LeagueEnv`).** The league-enabled layer inserts opponent
  selection between physics and the task. A league entry contains a stable
  opponent identifier, a category, and a frozen complete hierarchical parameter
  tree. The fixed league supports controlled warm-start experiments. The
  historical league stores a nonempty immutable population and selects an entry
  deterministically from the rollout seed. Opponent parameters are never
  updated by the learner's optimizer, so each rollout compares the learner to a
  stationary policy even though the learner changes between updates.

- **TAG task (`MultiAgentTaskEnv`).** The task layer assigns exactly one pursuer
  and one evader, computes each agent's opponent-relative planar position and
  velocity in its own heading frame, and reports the arena center in the same
  local frame. It detects inter-agent tag contact, persistent falls, arena exits,
  non-finite state, and the ten-second episode timeout. The same task class is
  used for league training rather than maintaining a second parallel task
  implementation.

- **Shared role-conditioned strategy policy.** A single parameter set represents
  both pursuer and evader behavior. Its 11-dimensional observation concatenates
  local opponent position (2), local opponent velocity (2), local arena-center
  position (2), the previous high-level command (3), and a two-element one-hot
  role encoding. Position, velocity, and command fields are normalized by fixed
  physical scales. This explicit role input allows one network to implement two
  different game objectives while sharing useful representations.

- **Strategy encoder and head.** The strategy encoder is a two-layer multilayer
  perceptron with 128 tanh units per layer. Its stochastic actor head parameterizes
  a diagonal Gaussian over three latent action coordinates; tanh squashing and
  elementwise scaling convert samples into the interpretable command
  `(forward_velocity, lateral_velocity, yaw_rate)` with limits `(1.0, 0.5, 1.0)`.
  A separate scalar value head estimates the role-conditioned strategic return.
  The strategy is evaluated at 10 Hz, so each command is held across five 50 Hz
  locomotion decisions.

- **Boundary-safe command execution.** PPO retains the raw sampled strategy
  action and its exact log probability for optimization. Before physical
  execution, a deterministic safety projection suppresses command components
  that would drive an agent farther outside the configured boundary margin;
  tangential motion and yaw remain available. Separating the sampled action from
  the executed command avoids corrupting PPO's likelihood ratio while preventing
  a learned outward command from becoming an avoidable arena termination.

- **Locomotion observation and controller.** The motor policy receives the
  accepted 103-dimensional P2 observation contract: local root linear velocity
  (3), root angular velocity (3), projected gravity (3), the current strategy
  command (3), joint displacement from the default pose (29), joint velocity
  (29), previous motor action (29), and sine/cosine gait phase for the two legs
  (4). It receives no role bit, opponent identity, or opponent geometry. This
  information boundary forces the strategy module to decide *where to move* and
  the locomotion module to solve only *how to execute that motion physically*.

- **Locomotion encoder and motor head.** The locomotion encoder is a
  512-256-128 SiLU multilayer perceptron. Its diagonal-Gaussian actor produces 29
  tanh-bounded motor actions, which are scaled into joint-position targets, and
  its scalar value head predicts locomotion return. Observation normalization,
  encoder weights, and compatible actor weights are migrated from the accepted
  40-million-step P2 controller. The hierarchy therefore begins with a capable
  locomotion prior instead of relearning standing and walking from random
  initialization.

- **Explicit hierarchy and gradient boundary.** The strategy command is passed
  through `stop_gradient` before it is concatenated with the locomotion
  observation. Strategy and locomotion parameters occupy separate named
  checkpoint subtrees. Consequently, locomotion loss cannot change strategic
  parameters through the command channel, and strategy loss cannot directly
  alter the motor network. The modules are optimized during the same training
  run but remain independently attributable and can use different data cadence,
  learning rates, and acceptance rules.

- **Strategic objective.** Strategy PPO uses role-selected TAG reward. Pursuer
  return rewards reducing opponent distance and completing a tag; evader return
  rewards increasing or maintaining separation and surviving until timeout.
  Both include task safety and command-regularity terms. Strategy transitions
  occur only when a new 10 Hz command is sampled, and their GAE/value targets are
  computed independently from the locomotion targets.

- **Locomotion objective.** Locomotion PPO optimizes command execution and
  physical stability rather than the game outcome. Its reward combines
  exponential planar-velocity tracking, yaw-rate tracking, uprightness, action
  smoothness, joint-deviation regularization, and a fall penalty. The strategy
  may discover difficult task motions, while this separate objective determines
  whether the low-level controller can execute them safely.

- **Independent PPO updates.** The strategy optimizer uses Adam at `3e-4` on
  every iteration. The locomotion optimizer uses Adam at `1e-5` every fifth
  iteration. Both clip the global gradient norm at 1.0. A scheduled locomotion
  update is rejected if its pre-update approximate KL is non-finite or exceeds
  0.01, protecting the pretrained motor prior from destructive policy jumps.
  Optimizer states, update counters, parameter trees, and configuration are all
  restored exactly when training resumes.

- **Balanced-role rollout batch.** Each update uses 16 parallel environments:
  eight collect learner-as-pursuer trajectories and eight collect
  learner-as-evader trajectories. The two halves face the same sampled frozen
  opponent checkpoint, run for a complete 500-step, ten-second episode horizon,
  compute role-specific advantages, and are flattened into one strategy batch
  and one locomotion batch. This exact 50/50 construction prevents stochastic
  runs of one role from moving the shared strategy network far enough to forget
  the other role.

- **Historical self-play.** The current league population contains accepted
  hierarchical checkpoints from iterations 10, 15, and 20. It therefore exposes
  each learner seed to policies from multiple stages of training instead of only
  its own latest behavior. This is a first stationary historical-league
  experiment, not yet a dynamically growing population or a learned matchmaking
  system.

- **Checkpoint evaluation and selection.** Evaluation runs each checkpoint in
  both roles across deterministic held-out seeds. Selection is lexicographic:
  minimum role success, minimum physical survival, mean role success, mean
  survival, mean pursuer tag time, worst-role tracking RMSE, and finally earlier
  iteration. The ordering prevents aggregate reward or one strong role from
  hiding role collapse, locomotion failure, or reduced safety.

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

## Boundary-safe hierarchical execution

The hierarchy now distinguishes the raw strategy action from the command sent to
the locomotion controller. `strategy_actions` retains the exact sampled action
used to compute PPO log probability. `commands` contains its executed form. In
the final one-meter arena margin, outward world-frame velocity is attenuated
linearly to zero at the boundary while inward velocity, tangential velocity, and
yaw remain available. This preserves PPO's action-distribution semantics while
adding a hard execution constraint.

Applying the constraint only during evaluation of the previously trained
iteration-50 checkpoint raised evader survival from 0% to 100% and pursuer
survival from 0% to 50%. Because that policy had never trained through the
projection, a matched 80,000-step run was then trained with safe execution active
throughout. It retained exact 50/50 role sampling, completed ten locomotion and 50
strategy updates, rejected no locomotion updates, and ended with locomotion KL
of `1.9e-7`.

| Iteration 50 variant | Pursuer success/survival | Evader success/survival |
| --- | ---: | ---: |
| balanced, no safety | 0% / 0% | 0% / 0% |
| old checkpoint, safety at evaluation only | 0% / 50% | 100% / 100% |
| safety active during training and evaluation | 25% / 75% | 100% / 100% |

The constraint fixes the evader's boundary failure and substantially improves
pursuer survival. It does not yet produce a strong late pursuer: the retrained
pursuer tagged in only one of four evaluation seeds. The remaining issue is
therefore strategy/curriculum quality rather than role omission, low-level KL
collapse, or unconstrained boundary commands.

## Role-conservative checkpoint selection

Checkpoint selection is now an explicit, reproducible evaluation stage. Each
candidate must contain valid pursuer and evader evaluations. Candidates are
ranked lexicographically by minimum role success, minimum physical survival,
mean role success, mean survival, and finally worst-role command-tracking RMSE.
An exact tie favors the earlier checkpoint. This ordering prevents a strong role
from hiding regression in the other role and treats locomotion safety as more
important than tracking quality.

All five checkpoints from the safety-aware 80,000-step run were evaluated over
four seeds per role. Iterations 10, 20, 30, and 40 each achieved 100% role
success and 100% physical survival for both roles. Iteration 10 was selected
because its worst-role tracking RMSE (`0.131`) was lower than iteration 20
(`0.164`), iteration 30 (`0.187`), and iteration 40 (`0.210`). Iteration 50
ranked last because minimum role success fell to 25% and minimum physical
survival to 75%.

The selected checkpoint is therefore iteration 10, not the final iteration.
The complete audit record is written to
`logs/hierarchical/evaluation/tag_joint_balanced_safe_80k_selection.json`.

## Full-episode TAG curriculum

The original hierarchical launcher collected 100 control steps per update,
equivalent to two seconds, and reset every environment before the next update.
The selected pursuer requires roughly 5.3 seconds to tag in deterministic
evaluation. Consequently, the short training segments could not contain the
pursuer's terminal tag bonus or the evader's timeout reward. PPO optimized only
the dense shaping terms and bootstrapped value estimates, which explains why
long-horizon behavior could regress while short training metrics remained
finite.

The launcher now defaults to 500 control steps and a 10-second episode. It
requires the rollout duration to equal the episode duration, so every on-policy
batch can represent tag, timeout, fall, or boundary termination. Training also
logs role-specific pursuer tag and evader timeout rates instead of relying only
on aggregate reward.

A controlled 80,000-step continuation resumed the selected iteration-10
checkpoint and changed only the rollout/episode horizon. After the first update,
all remaining batches recorded a 100% pursuer tag rate and 100% evader timeout
rate. The iteration-20 checkpoint then achieved 100% success and 100% physical
survival for both roles over four deterministic evaluation seeds. Mean pursuer
tag time fell from approximately 5.3 seconds to approximately 4.0 seconds. Mean
tracking RMSE was `0.144` for the pursuer and `0.124` for the evader.

This establishes full-episode collection as the fixed-opponent TAG curriculum
baseline. The W&B run is `og63hoc3`, and the held-out report is
`logs/hierarchical/evaluation/tag_joint_balanced_safe_full_episode_iter020.json`.

A larger matched evaluation then used 16 held-out seeds (4--19) for the parent
iteration 10 and full-episode iterations 15 and 20. Every checkpoint retained
100% role success and 100% physical survival for both roles. Mean pursuer tag
time improved from `3.14 s` at the parent to `2.71 s` at iteration 15 and
`2.51 s` at iteration 20. The full-episode curriculum therefore retained both
roles while making pursuit faster across the expanded seed set.

The frozen conservative selector still chooses the parent iteration 10 because
all outcome and survival fields tie, after which its worst-role tracking RMSE
of `0.141` beats `0.148` at iteration 15 and `0.166` at iteration 20. This is a
deliberate application of the predeclared rule rather than post-hoc selection
for faster tags. The matched selection audit is stored in
`logs/hierarchical/evaluation/tag_joint_balanced_safe_full_episode_selection_heldout16.json`.

For subsequent experiments, the checkpoint rule was prospectively extended to
use mean pursuer tag time after role success and physical survival, but before
tracking RMSE. Failed pursuer rollouts are charged the complete evaluation
horizon, so early physical failure can never appear to be a fast tag. This
retains the conservative role and safety gates while distinguishing policies
that solve TAG more efficiently. A regression fixture freezes this ordering and
verifies that tag speed cannot override weaker role success or survival.

Under the revised rule, iteration 20 ranks first, followed by iteration 15 and
the parent iteration 10. Iteration 20 is therefore the accepted checkpoint for
the next phase. Its selection audit is
`logs/hierarchical/evaluation/tag_joint_balanced_safe_full_episode_selection_tag_time_heldout16.json`.

## Historical hierarchical league

The league layer now supports a nonempty immutable population of historical
hierarchical policies. Every entry has a unique opponent ID, the `historical`
category, and a complete frozen strategy-plus-locomotion parameter tree.
Selection is deterministic from the rollout seed and cycles uniformly over the
registered population. Invalid categories and duplicate identities are rejected.

The training launcher accepts a list of hierarchical checkpoint directories,
validates each checkpoint digest, restores its policy parameters, and records
the resulting opponent IDs in the run manifest. Balanced pursuer and evader
halves use the same sampled opponent within an update, so the single recorded
opponent identity accurately describes the complete batch. Successive updates
advance through the historical pool by seed.

The first integration smoke restored accepted iteration 20 as the learner and
registered iterations 10, 15, and 20 as historical opponents. One complete
10-second, 16-environment update remained finite, preserved the 50/50 role
split, applied both strategy and locomotion updates, and saved iteration 21.
The sampled opponent was historical iteration 20 with its checkpoint digest in
the recorded ID. Pursuer tag rate was 100%; evader timeout rate was 0% against
the stronger historical pursuer, providing a concrete learning signal for the
next longer run. Evidence is stored under
`logs/hierarchical/training/tag_historical_league_smoke_seed0`.

## Planned scaled experiment matrix

The available training system has 16 Blackwell GPUs. Once the diagnostic
curricula and checkpoint-selection rule are frozen, the main experiment can run
four independent seeds for each of four conditions concurrently:

1. TAG curriculum;
2. aggressive-command curriculum;
3. terrain curriculum;
4. random-command baseline.

This yields 16 independent runs rather than using multiple devices to accelerate
one seed. The design supports seed-level uncertainty estimates and direct
curriculum ablations while keeping each optimizer trajectory independent.

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

1. Run a longer multi-seed TAG pilot against the frozen historical pool and
   evaluate intermediate checkpoints conservatively.
2. Add controlled mixtures of historical, recent, and best opponents after the
   static historical pilot is stable.
3. Use TAG failure trajectories to identify difficult locomotion states for
   targeted low-level training and ablation studies.

The current evidence does not yet justify claiming a converged TAG policy. It
does justify that the hierarchical execution, optimization, checkpointing,
evaluation, contact capacity, locomotion stability, and balanced-role data path
are implemented and individually validated.
