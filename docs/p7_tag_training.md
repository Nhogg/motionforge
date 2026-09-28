# P7 TAG training

P7 trains one high-level role at a time while keeping the opponent and the
accepted P2 locomotion controller fixed. This preserves a stationary opponent
for the first learning problem and keeps tag strategy separate from low-level
joint control.

## Frozen scripted evader

`FrozenScriptedEvader` is the opponent boundary for pursuer training. It binds
the existing boundary-aware scripted evader to the environment's evader agent
index and exposes a single observation-to-command operation. Its dataclass and
nested configuration are immutable. A canonical, versioned JSON specification
is hashed with SHA-256 and will be recorded in every pursuer experiment, so a
changed gain, limit, role assignment, or specification version produces a
different opponent identity.

The frozen object contains no learned parameters and owns neither locomotion
inference nor physics. The accepted P2 checkpoint remains a separate fixed
dependency that maps both agents' velocity commands to joint targets. This
keeps opponent policy, controller, and environment modular.

Evidence: `logs/p7/tag_frozen_evader_a.json` and
`logs/p7/tag_frozen_evader_swapped_a.json`. Both role assignments passed on the
GPU backend. The audit verifies immutable and versioned configuration,
deterministic JIT commands, finite bounded output, correct evader-role binding,
stable fingerprints for identical specifications, and changed fingerprints
when an evasion gain changes. The canonical agent-1 opponent fingerprint for
the initial pursuer experiment is
`4e973d5c79f8e2e1a0de8b44189a7323cd6d9f9aec5f8985671f0668b3ce91b6`.

## Initial pursuer learning contract

The first learned pursuer observes nine normalized values: relative planar
position and velocity of the evader, the local direction to arena center, and
the previous pursuer command. Episode time is deliberately excluded so the
policy cannot key its strategy to the fixed timeout. Its three normalized
actions map to the accepted `[1.0, 0.5, 1.0]` forward, lateral, and yaw-rate
limits and are held for five 50 Hz locomotion updates, producing a 10 Hz
high-level decision rate.

The dense reward is decomposed into distance progress, a small proximity term,
a per-decision cost, and a weak command-change penalty. Physical tag contact
earns the terminal bonus. Only pursuer falls and pursuer boundary violations
receive failure penalties; an evader failure terminates through the game rules
but is not mislabeled as a successful tag. Timeout is exposed to Brax as a
truncation so value bootstrapping can distinguish it from physical terminal
events. Every component is retained as a separate metric.

The initial PPO function approximator uses separate actor and critic MLPs.
Each has two 128-unit hidden layers with `tanh` activation. The actor supplies
a three-dimensional tanh-normal action distribution, which bounds sampled
commands before the environment applies physical command scaling. The critic
receives the same nine-value observation but shares no parameters with the
actor.

Evidence: `logs/p7/tag_pursuer_environment_wrapped_a.json` and
`logs/p7/tag_pursuer_networks_b.json`. The environment audit passed on the GPU
backend, including nine-value clock-free observations, exact action scaling,
five-update command holding, finite integrated physics, independent reward
fixtures, timeout truncation semantics, batched wrapper bookkeeping, and
MJX-Warp-safe selective auto-reset. The network audit initialized and executed
finite batched inference with the expected shapes and bounded sampled actions.
The actor has 18,566 parameters and the separate critic has 17,921.

`scripts/training/train_tag_pursuer.py` is the reproducible PPO entry point. It
records the frozen evader specification and fingerprint, resolved locomotion
checkpoint, reward/environment configuration, network architecture,
dependencies, repository state, device list, PPO parameters, JSONL progress,
checkpoints, summary checks, and optional W&B scalar metrics. Its custom Brax
episode wrapper preserves task timeout truncations so PPO can bootstrap them;
all resettable controller memory lives in pipeline state to prevent leakage
between auto-reset episodes. The auto-reset wrapper uses MJX `Data.where`
through a per-environment map because generic tree selection corrupts Warp's
non-batched contact metadata.

The end-to-end GPU smoke run is recorded under
`logs/p7/training/tag_pursuer_smoke_f`. It completed 4,480 environment steps,
wrote checkpoint `000000004480`, recorded finite training/evaluation metrics,
preserved the frozen evader fingerprint, and passed every launcher check. This
is an integration audit, not evidence that the pursuer has learned TAG; the
first substantive training run remains the next P7 experiment.

## Initial pursuer pilot

The seed-0 pilot in `logs/p7/training/tag_pursuer_128k_seed0` reached 163,840
environment steps with finite metrics and four checkpoints. Its training-time
evaluation briefly reported 62.5% tag success at checkpoint `000000122880`,
but this did not generalize. The dedicated deterministic evaluator
`scripts/evaluation/evaluate_tag_pursuer.py` restored that checkpoint and ran
32 unseen reset seeds (`1000` through `1031`). It produced zero tags and 32
pursuer boundary exits, with no falls or non-finite states. Evidence is stored
in `logs/p7/tag_pursuer_122880_heldout_32.json`.

This rejects longer training with the current reward rather than completing
the pursuer-training milestone. The next P7 iteration should add dense arena
boundary shaping and repeat the bounded pilot before increasing the training
budget.

The repair adds a quadratic boundary-margin penalty without changing the
observation, action, network, opponent, or PPO configuration. The penalty is
zero throughout the central 6-by-6-meter square, rises smoothly through the
outer one-meter margin, and reaches `-1.0` per high-level decision at the
4-meter arena edge. The existing `-10.0` terminal boundary penalty remains in
place. This gives PPO advance warning before an otherwise sparse boundary
failure while preserving unrestricted pursuit through most of the arena.

The matched seed-0 boundary pilot is stored under
`logs/p7/training/tag_pursuer_boundary_128k_seed0`. It reached 163,840
environment steps and repeatedly achieved 100% tag success with no boundary
exits on the fixed training-time evaluator. That apparent improvement did not
transfer to unseen resets. Deterministic evaluation of both checkpoint
`000000081920` and the final checkpoint `000000163840` on 32 held-out seeds
produced zero tags and 32 pursuer boundary exits. The final checkpoint's mean
minimum separation was 1.490 meters and its mean episode length was 67.97
high-level decisions. Evidence is stored in
`logs/p7/tag_pursuer_boundary_81920_heldout_32.json` and
`logs/p7/tag_pursuer_boundary_163840_heldout_32.json`.

This rejects boundary-penalty strength as the immediate bottleneck: the policy
can optimize the fixed evaluator but does not learn a state-conditioned
boundary response that generalizes. The next bounded P7 experiment should
address reset coverage. The current auto-reset path returns each vectorized
environment to its cached initial pipeline state, so an environment slot sees
the same spawn geometry on every episode. The next repair should sample a fresh
reproducible reset state after every terminal episode, then repeat this pilot
and held-out evaluation before changing the network or increasing the training
budget.

## Fresh-reset experiment

The training wrapper now owns a separate per-environment PRNG stream and draws
a new reset state after every terminal episode. Reset selection still uses MJX
`Data.where` for Warp-safe batched state merging. Identical initial seeds replay
the same reset sequence, while consecutive episodes no longer reuse a cached
spawn geometry. The GPU audit is recorded in
`logs/p7/tag_pursuer_environment_fresh_reset_a.json`.

The matched seed-0 pilot in
`logs/p7/training/tag_pursuer_fresh_reset_128k_seed0` reached 163,840
environment steps with finite metrics and four checkpoints. Checkpoint
`000000081920` was selected because its fixed 16-environment evaluation had
100% tag success. On the same 32 held-out seeds used by the earlier pilots, it
produced zero tags and 32 pursuer boundary exits. Its mean minimum separation
was 1.987 meters, essentially unchanged from the 2-meter initial separation,
and its mean episode length was 84.22 high-level decisions. Evidence is stored
in `logs/p7/tag_pursuer_fresh_reset_81920_heldout_32.json`.

This rejects cached auto-reset geometry as the primary cause of the evaluation
gap. Longer training is not justified yet. The next P7 diagnostic should audit
the fixed Brax evaluator and standalone held-out evaluator end to end using the
same checkpoint and explicit reset keys, including normalized observations and
deterministic actions. That comparison must establish why a policy reported as
successful by the training evaluator barely reduces separation under the
standalone evaluator before another reward, architecture, or training-budget
change is attempted.

## Deterministic evaluation and accepted pursuer

The parity audit found that Brax PPO's training evaluator was using its default
stochastic policy while the standalone evaluator correctly used the
deterministic policy mean. Wrapped and standalone observations, normalization,
and deterministic actions match exactly for identical reset keys; evidence is
stored in `logs/p7/tag_pursuer_evaluator_parity_a.json`. Training now sets
`deterministic_eval=true` explicitly, so checkpoint selection measures the
deployable policy rather than favorable exploration noise.

Two matched 128K diagnostics confirmed the mechanism. Reducing only the
entropy coefficient to `0.001`, and then initializing the action standard
deviation at `0.2` with zero entropy cost, did not work: the learned standard
deviation returned to approximately `0.9` and every deterministic evaluation
failed. For the tanh-normal Brax policy, the initial-noise setting does not
constrain the learned scale.

The accepted actor therefore learns only the three action means and appends a
fixed tanh-normal exploration standard deviation of `0.2`. This preserves
controlled stochastic exploration during PPO rollouts but prevents variance
from becoming a substitute for a useful deterministic policy. The network
audit verifies the exact fixed scale in
`logs/p7/tag_pursuer_networks_fixed_noise_a.json`. A matched 128K pilot became
safe but did not yet tag: all 32 held-out episodes timed out, with mean minimum
separation improving to 1.739 meters. This justified increasing training time
without another reward change.

The accepted seed-0 run is
`logs/p7/training/tag_pursuer_fixed_noise_1m_seed0`. It reached 1,146,880
environment steps with finite metrics and eight checkpoints. Deterministic tag
success first appeared at checkpoint `000000430080`, temporarily regressed,
and returned at the final checkpoint `000001146880`; this makes checkpoint
selection necessary even with fixed exploration. The final fixed evaluator
reported 100% tags, zero failures, and a mean episode length of 51 decisions.

The final checkpoint then tagged the frozen scripted evader on all 32 held-out
reset seeds (`1000` through `1031`) with zero falls, boundary exits, or
timeouts. Mean tag time was 51.78 high-level decisions, mean minimum separation
was 0.312 meters, and mean cumulative reward was 13.60. The accepted
machine-readable report is
`logs/p7/tag_pursuer_fixed_noise_1146880_heldout_32_b.json`. The evaluator's
repeatability check compares terminal outcome rather than exact terminal step,
because GPU contact resolution can move an otherwise identical tag by one
simulation step. The standalone evaluator resolves the fixed-noise network
contract from the run manifest beside the checkpoint unless explicitly
overridden. These results complete flat-ground pursuer training and
consistent pursuit verification for P7; the next phase-order item is freezing
this pursuer and training an evader.

## Frozen learned pursuer and evader environment

The accepted pursuer is restored from checkpoint
`logs/p7/training/tag_pursuer_fixed_noise_1m_seed0/checkpoints/000001146880`
as an immutable deterministic opponent. Its versioned specification records
the agent index, observation contract, fixed action-noise standard deviation,
and a digest of the checkpoint contents. This prevents an evader experiment
from silently changing its opponent while retaining the same nominal path.

The initial evader uses the same nine-value local observation contract and
three normalized velocity-command actions as the pursuer. Its reward reverses
distance progress, adds a small survival reward, penalizes proximity, abrupt
command changes, and boundary intrusion, and assigns explicit terminal values:
timeout is success; tag, evader fall, and evader out-of-bounds are failures.
Timeout is deliberately a true task terminal rather than a PPO truncation.
The frozen pursuer receives its own previous-command observation and is
evaluated once per high-level action, with its command held through the same
five low-level control updates as the evader command.

Evidence: `logs/p7/tag_frozen_pursuer_a.json` records frozen-opponent
fingerprint `72c878e5d862537d10095842e96c67b5e8c1c3def93a79549d3f4708cda52663`.
`logs/p7/tag_evader_environment_a.json` passed all 13 GPU checks, including
role binding, frozen-pursuer integration, observation memory, reward signs,
action repeat, finite physics, and batched fresh reset.

The first evader PPO smoke run is
`logs/p7/training/tag_evader_fixed_noise_128k_seed0`. It reached 143,360
environment steps with finite metrics, preserved the frozen pursuer
fingerprint, and maintained the exact `0.2` exploration standard deviation.
Deterministic evaluation still ended in tags for all 16 environments after 29
high-level decisions, with no evader falls or boundary exits. This validates
the training path but is not evidence of learned evasion; the next bounded
experiment increases training to approximately one million environment steps
without changing the reward contract.

The one-million-step run is
`logs/p7/training/tag_evader_fixed_noise_1m_seed0_retry` (W&B run
`g9drow6d`). It reached 1,146,880 environment steps with finite metrics and
seven checkpoints while preserving the frozen pursuer identity. Deterministic
evaluation was non-monotonic: checkpoints `000000327680` and `000000983040`
timed out the pursuer in all 16 evaluation environments, while intervening
checkpoints were tagged and the final checkpoint regressed to tags at 194 of
200 decisions. Checkpoint `000000983040` is therefore the provisional evader
candidate, but it requires a separate held-out-seed evaluation before the
freeze-and-train checklist item can be completed. The aborted 32-evaluation-
environment attempt is retained at
`logs/p7/training/tag_evader_fixed_noise_1m_seed0`; it was stopped during an
excessively expensive compile and is not an experiment result.

Independent deterministic held-out evaluation selected checkpoint
`000000327680`. It timed out the accepted frozen pursuer on all 32 seeds from
1000 through 1031, with zero tags, falls, boundary exits, or incomplete
episodes. Mean minimum separation was 0.865 meters and mean cumulative reward
was 7.36. The later `000000983040` candidate was weaker: it timed out on 31 of
32 seeds and was tagged on seed 1029. The accepted machine-readable report is
`logs/p7/tag_evader_327680_heldout_32.json`; the comparison report is
`logs/p7/tag_evader_983040_heldout_32.json`. The evaluator now requires every
rollout to time out for its acceptance check. These results complete the P7
freeze-pursuer-and-train-evader item. The next phase-order item is comparing
the learned roles' state distributions against scripted and random-command
locomotion before entering P8 self-play.

For that comparison, the accepted evader is restored through a role-specific
immutable wrapper rather than being mislabeled as a pursuer merely because the
two policies share a network shape. Its specification has checkpoint digest
`a227dbd40b0586fee5b3bf3b6495a2e7c3f3623cf8d4dbe25a13f1664e0bf896` and
fingerprint `c7c1b38f667afd9d7b455e2a480ecf25b49f8ee635fea869730a0f274bc00399`.
Evidence: `logs/p7/tag_frozen_evader_learned_a.json`; all ten deterministic,
immutability, bounds, provenance, and GPU checks passed.

## Shared state-distribution extraction

The final P7 comparison begins from a policy-agnostic raw extraction boundary
rather than separate learned, scripted, and random-policy log formats.
`motionforge/datasets/tag_state.py` defines schema version 1 and records both
agents' root poses, generalized root velocities, joint positions and
velocities, high-level commands, low-level normalized actions and joint
targets, opponent-relative observations, arena-center observations, separation,
root stability, near-fall flags, bounds state, tag contacts, and terminal task
flags. It performs no inference, simulation, host transfer, or serialization;
those responsibilities remain with the future condition-specific generator.

Evidence: `logs/p7/tag_state_extractor_a.json`. The GPU smoke test passed all
11 checks for exact shapes, command round-trip, finite values, initial
stability, reset separation, and schema identity. The next step is generating
small equal-budget learned, scripted, and random-command datasets through this
single extractor before scaling the comparison.
