# MotionForge Hierarchical Architecture Refactor

## Goal

Refactor MotionForge from the current architecture:

```text
separate pursuer policy
        or
separate evader policy
          |
          v
    [vx, vy, wz]
          |
          v
frozen pretrained locomotion controller
          |
          v
    29 joint actions
```

into a **single role-conditioned hierarchical policy** with:

```text
strategic observations + role
          |
          v
   strategy encoder
          |
          v
    strategy head
          |
          v
 high-level command z
          |
     stop_gradient
          |
          v
proprioception + motor state + z
          |
          v
 locomotion encoder
          |
          v
     motor head
          |
          v
   29 joint actions
```

Both the strategic and locomotion portions must ultimately be trainable, but they must have **separate inputs, separate representations, separate losses, and an explicit gradient boundary**.

The hierarchy should communicate through a compact high-level command. For the first implementation, retain the existing interpretable command:

```text
z = [vx, vy, yaw_rate]
```

Do **not** replace this with a learned latent yet.

The architecture must additionally support:

1. one shared role-conditioned policy for pursuer and evader;
2. training experience from both roles;
3. historical/league self-play;
4. frozen opponent snapshots during rollout segments;
5. independent strategic and locomotion optimization;
6. later terrain conditioning without redesigning the core architecture.

---

# 1. Preserve the working system before refactoring

Do not delete the current P7 implementation.

The existing architecture is a valid baseline and should remain reproducible.

Before changing behavior:

- preserve current pursuer checkpoint loading;
- preserve current evader checkpoint loading;
- preserve deterministic evaluation;
- preserve current two-G1 physics environment;
- preserve existing reward calculations;
- preserve existing low-level G1 checkpoint;
- preserve current scripted policies;
- preserve existing P2/P3/P4/P7 tests;
- preserve old configs where practical.

Create the new architecture alongside the existing implementation until parity tests pass.

Do not silently rewrite historical experiment behavior.

The current sequential pursuer/evader implementation should remain available as a baseline.

---

# 2. Terminology

Use these names consistently.

## Strategic level

Responsible for:

- pursuit/evasion strategy;
- opponent reasoning;
- terrain-aware navigation later;
- arena reasoning;
- role conditioning;
- producing high-level velocity commands.

Call its components:

```text
StrategyEncoder
StrategyHead
```

Avoid calling this simply the "high-level head" if doing so obscures that it has its own encoder.

## Locomotion level

Responsible for:

- executing the high-level command;
- balance;
- gait;
- contacts;
- joint control;
- recovery;
- whole-body motion.

Call its components:

```text
LocomotionEncoder
MotorHead
```

## Full model

Call the complete network something like:

```text
HierarchicalPolicy
```

or:

```text
MotionForgePolicy
```

Do not describe the design as merely "two heads."

The two modules are **sequentially conditioned**, not two parallel heads on a common trunk.

---

# 3. Full model architecture

Implement a model conceptually equivalent to:

```python
class HierarchicalPolicy:
    strategy_encoder
    strategy_head

    locomotion_encoder
    motor_head
```

Forward data flow:

```text
role -------------------------------+
                                     |
opponent state ----+                 |
arena state -------+--> StrategyEncoder
terrain state -----+                 |
history -----------+                 |
                                     v
                              strategy features
                                     |
                                     v
                               StrategyHead
                                     |
                                     v
                             [vx, vy, yaw_rate]
                                     |
                              stop_gradient
                                     |
                                     +----------------------+
                                                            |
proprioception ---------------------------------------------+
contacts ---------------------------------------------------+
gait state -------------------------------------------------+
previous motor action --------------------------------------+
                                                            |
                                                            v
                                                  LocomotionEncoder
                                                            |
                                                            v
                                                       MotorHead
                                                            |
                                                            v
                                                   29 motor actions
```

For the first version, the strategy output remains:

```python
command.shape == (3,)
```

with:

```text
command[0] = desired forward pelvis-frame velocity
command[1] = desired lateral pelvis-frame velocity
command[2] = desired yaw rate
```

Retain the current physical command limits unless there is an explicit configuration override.

---

# 4. Separate observations for the two hierarchy levels

Do **not** concatenate all state and feed it through one encoder.

Create explicit observation schemas.

## Strategy observation

The strategy policy should consume task-level information.

Initial fields:

```text
relative opponent planar position
relative opponent planar velocity
arena-center vector / boundary context
previous high-level command
role
```

Prepare the schema for future addition of:

```text
terrain features
short history / recurrent state
```

but do not implement unnecessary terrain features until terrain training begins.

Define an explicit type, for example:

```python
@dataclass
class StrategyObservation:
    opponent_position_local: Array
    opponent_velocity_local: Array
    arena_center_local: Array
    previous_command: Array
    role: Array
```

The representation should be array/JAX-friendly.

Do not allow strategy code to access arbitrary raw `qpos`/`qvel`.

## Locomotion observation

The locomotion module should consume motor-relevant information.

Use the existing P2 locomotion observation contract as the starting point.

It should contain the state needed for locomotion such as:

```text
pelvis/root velocity
angular velocity / gyro
projected gravity
joint positions
joint velocities
previous motor action
gait phase
high-level command z
```

Add contact information only if it can be integrated cleanly and there is a specific motor-training reason to use it.

Do not feed opponent identity or opponent geometry directly into the locomotion encoder.

The locomotion module should know:

> execute this motion request.

It should not know:

> I am chasing this opponent.

That separation is intentional.

---

# 5. Role conditioning

Replace separately architected pursuer and evader networks with **one strategic policy conditioned on role**.

Represent role explicitly.

For the initial implementation use either:

```text
[1, 0] = pursuer
[0, 1] = evader
```

or an integer followed by a tiny embedding.

Prefer the simplest implementation unless a learned embedding materially simplifies the existing network API.

The strategic policy becomes:

```text
pi_strategy(command | observation, role)
```

not:

```text
pi_pursuer(...)
pi_evader(...)
```

The same parameter set must be capable of acting as either role.

---

# 6. Role assignment during training

A training segment must support the learner occupying either role.

Initially sample:

```text
P(learner is pursuer) = 0.5
P(learner is evader) = 0.5
```

Make this configurable.

Example configuration:

```yaml
roles:
  pursuer_probability: 0.5
```

For every environment/episode:

```text
learner role = sampled role
opponent role = opposite role
```

The learner always receives the correct role conditioning.

The frozen opponent must also receive the appropriate opposite role conditioning.

Do not maintain separate pursuer and evader model architectures.

---

# 7. Gradient boundary

Implement an explicit stop-gradient between strategic output and locomotion input.

Conceptually:

```python
command = strategy_head(strategy_features)

motor_command = jax.lax.stop_gradient(command)

motor_features = locomotion_encoder(
    proprioception,
    motor_state,
    motor_command,
)

motor_action = motor_head(motor_features)
```

The purpose is:

```text
strategy objective
    updates strategy encoder + strategy head

locomotion objective
    updates locomotion encoder + motor head
```

while preventing:

```text
locomotion loss
    ->
strategy network
```

The command itself still affects motor behavior.

Stop-gradient means **optimization is decoupled**, not that the modules do not communicate.

Add a unit test proving this.

For a synthetic loss depending only on the motor objective:

```text
grad(strategy parameters) == 0
grad(locomotion parameters) != 0
```

within numerical tolerance.

This test is required.

---

# 8. Strategy training objective

Continue using PPO for the strategic policy initially.

The strategic PPO objective should receive the game reward associated with the learner's current role.

Pursuer and evader reward semantics may remain different, but reward selection should be role-conditioned rather than implemented by completely separate training architectures.

Conceptually:

```python
reward = tag_reward(
    state,
    role=learner_role,
)
```

rather than:

```python
if training_pursuer_program:
    ...
elif training_evader_program:
    ...
```

Retain the useful existing reward behavior where possible.

Do not accidentally treat an opponent fall or OOB event as a successful tag unless explicitly intended.

Preserve correct timeout semantics per role.

---

# 9. Locomotion training objective

The locomotion module must no longer be permanently frozen.

However, do not simply propagate TAG reward through the motor controller.

Implement a separate locomotion loss/reward objective.

The locomotion objective should remain focused on physically executing the requested command while maintaining stable locomotion.

Start from the existing Playground locomotion reward formulation where practical.

The objective should include terms corresponding to:

```text
velocity command tracking
yaw-rate tracking
uprightness/stability
joint/action regularization
smoothness
foot/contact behavior
fall avoidance
```

Reuse upstream definitions rather than inventing new losses if equivalent terms already exist.

The key distinction is:

```text
TAG/game reward
    -> strategic parameters

locomotion reward/loss
    -> motor parameters
```

Do not initially update motor parameters directly from the sparse tag win/loss signal.

---

# 10. Decide update cadence explicitly

Do not hide optimizer scheduling inside model code.

Expose a configuration determining when each level updates.

Initial recommended design:

```text
strategy rollout collection
       |
       +--> strategy PPO batch
       |
       +--> locomotion training samples
```

Then:

```text
strategy optimizer updates strategy parameters

locomotion optimizer updates motor parameters
using samples from the same physical experience
```

It is acceptable for the two optimizers to run during the same overall training iteration, but they must remain logically distinct.

Record metrics separately:

```text
strategy/loss
strategy/policy_loss
strategy/value_loss
strategy/reward
strategy/entropy

locomotion/loss
locomotion/tracking_error
locomotion/fall_rate
locomotion/command_error
...
```

Do not report one combined scalar `loss`.

---

# 11. Parameter tree

The checkpoint/parameter tree should make the hierarchy explicit.

Prefer something structurally similar to:

```python
params = {
    "strategy": {
        "encoder": ...,
        "head": ...,
    },
    "locomotion": {
        "encoder": ...,
        "motor_head": ...,
    },
}
```

If Brax/Flax constraints require a different exact structure, preserve equivalent semantic grouping.

This grouping must allow:

```python
strategy_params = ...
locomotion_params = ...
```

without manually searching parameter names.

This is important for:

- stop-gradient testing;
- separate optimizers;
- freezing/unfreezing;
- checkpoint migration;
- ablation experiments.

---

# 12. Initialize locomotion from the existing trained controller

Do **not** discard the accepted G1 locomotion prior.

Initialize the new locomotion module from the current accepted P2 checkpoint wherever architectures are compatible.

The current controller is already known to execute:

```text
[vx, vy, yaw_rate]
    ->
29 actions
```

so the new architecture should begin from that capability rather than relearning walking from zero.

If the exact module representation makes direct parameter loading impossible:

1. first reproduce the current architecture inside the new module;
2. load its existing parameters;
3. verify action parity;
4. only then add additional trainability or architectural modifications.

Create a regression test:

```text
same locomotion observation
same command
same checkpoint

old controller action ~= new locomotion module action
```

before beginning joint hierarchical training.

---

# 13. Do not share the strategy and locomotion encoder

The strategy encoder and locomotion encoder must be separate.

Do not implement:

```text
             shared encoder
              /        \
 strategy head       motor head
```

That is explicitly not the intended design.

The reason is that the two modules operate at different abstraction levels and consume different state.

The strategic representation should capture:

```text
opponent geometry
role
arena geometry
later terrain strategy
history
```

The motor representation should capture:

```text
proprioception
balance
joint dynamics
gait
contacts
command execution
```

Shared activations are not assumed to be useful.

---

# 14. Preserve the temporal hierarchy

Maintain:

```text
physics              500 Hz
locomotion policy     50 Hz
strategy policy       10 Hz
```

unless configuration explicitly changes it.

One strategic command should still be held across five locomotion decisions.

One locomotion action should still span ten physics steps.

Do not accidentally add another `action_repeat` in PPO that changes the effective strategic rate.

Add tests for the frequency relationship.

---

# 15. Environment ownership

Refactor responsibilities so environment layers remain clean.

Required environment inheritance structure:

```text
WarpEnv
    |
    v
LeagueEnv
    |
    v
MultiAgentTaskEnv
```

Do not create a separate parallel task-environment hierarchy for non-league
training. Evaluation, scripted play, and fixed-opponent training use the same
``MultiAgentTaskEnv`` with a league configured for the required fixed or
scripted opponent.

## WarpEnv owns

```text
MuJoCo/MJX state
G1 models
physics stepping
joint actions
contacts
terrain geometry
robot state extraction
```

WarpEnv must not know about:

```text
PPO
league snapshots
checkpoint ranking
```

## LeagueEnv owns

```text
League reference
opponent selection
snapshot identity
frozen-opponent lifecycle
matchup metadata
```

Conceptually:

```python
class LeagueEnv(WarpEnv):
    def __init__(self, ..., league: League):
        super().__init__(...)
        self.league = league
```

LeagueEnv must not own learner optimizer state. It may resolve an immutable
opponent snapshot, but policy execution remains part of rollout collection.

## MultiAgentTaskEnv owns

```text
role assignment
tag detection
game observations
game reward
timeout
OOB rules
termination semantics
```

MultiAgentTaskEnv should not own optimizer state or mutable learner weights.

## RolloutRunner owns

```text
learner policy execution
opponent policy execution
hierarchical inference
per-role routing
trajectory collection
```

## League owns

```text
policy snapshots
opponent sampling
opponent metadata
historical policy population
best-policy tracking
scripted opponents
```

Do **not** put neural-network weights inside ``WarpEnv``. Frozen opponent
snapshots belong to the league layer and trainable learner parameters belong to
the trainer.

---

# 16. League architecture

Implement league self-play separately from physics.

Conceptual interface:

```python
class League:
    def add_snapshot(...): ...
    def sample_opponent(...): ...
    def update_results(...): ...
```

Possible model:

```python
@dataclass
class PolicySnapshot:
    id: str
    step: int
    params: PyTree
    metrics: ...
    fingerprint: str
    source: Literal[
        "recent",
        "historical",
        "best",
        "scripted",
    ]
```

Do not require every opponent to have trainable weights; scripted opponents should satisfy the same acting interface.

---

# 17. League categories

Initial categories:

```text
recent
historical
best
scripted
```

Interpretation:

## Recent

Recent snapshots of the learner.

Purpose:

```text
train against behavior close to current capability
```

## Historical

Older snapshots sampled across training history.

Purpose:

```text
reduce forgetting
prevent cycling
preserve robustness to older strategies
```

## Best

One or a small set of snapshots selected according to a declared evaluation metric.

Purpose:

```text
maintain pressure from strongest known behavior
```

## Scripted

Existing deterministic pursuer/evader behaviors.

Purpose:

```text
anchor training
provide predictable opponent behavior
prevent complete drift
```

---

# 18. Initial opponent mixture

Use configurable probabilities.

Start with the architecture drawing's tentative distribution:

```yaml
league:
  recent_probability: 0.55
  historical_probability: 0.25
  best_probability: 0.10
  scripted_probability: 0.10
```

Do not hard-code these values.

Validate that probabilities sum to 1.

Log which category and which exact opponent snapshot was selected for every rollout.

This distribution is provisional and must be tunable without code changes.

---

# 19. Frozen opponent requirement

The sampled opponent must remain frozen for the relevant rollout/training collection segment.

Do not update the opponent copy during the learner's PPO optimization.

Training structure:

```text
current learner theta_t
         |
         +---------------------+
                               |
                               v
                        League.sample()
                               |
                               v
                   frozen opponent theta_k
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
       learner role                    opponent role
              |                                 |
              +------------- game --------------+
                            |
                            v
                         rollout
                            |
                            v
                    update theta_t only
```

After a configured interval:

```python
league.add_snapshot(current_params)
```

This is still self-play because the opponent population comes from the learner's own policy lineage.

---

# 20. Do not update both active copies from the same rollout initially

Avoid:

```text
pi_theta pursuer
vs
pi_theta evader

both optimized immediately from the same rollout
```

for the first implementation.

Instead:

```text
learner = current trainable policy
opponent = frozen league snapshot
```

The learner may be assigned either role.

This gives a substantially more stationary opponent during each PPO collection/update segment.

A future experiment may compare simultaneous self-play, but it should not be the initial implementation.

---

# 21. Snapshot timing

Expose:

```yaml
league:
  snapshot_interval_updates: N
  max_recent: ...
  max_historical: ...
```

Do not snapshot on every individual gradient update by default.

Create a snapshot only after a meaningful amount of learner training.

Fingerprint snapshots and record:

```text
Git revision
training step
parameter digest
role-conditioned architecture version
model configuration
parent lineage if useful
```

Reuse the project's existing opponent-fingerprinting discipline.

---

# 22. "Best" opponent selection

Do not define `best` as simply the latest checkpoint.

Track a declared evaluation score.

Possible initial measure:

```text
role-conditioned win rate across a fixed opponent/evaluation set
```

A policy only replaces the league's `best` snapshot if it beats the current best according to the declared metric.

Do not use training reward alone.

---

# 23. League evaluation/payoff matrix

Maintain matchup statistics.

At minimum track:

```text
learner snapshot
opponent snapshot
learner role
episodes
wins
losses
timeouts
falls
OOB
mean episode length
mean tag time
```

Build a payoff matrix conceptually like:

```text
             theta0 theta1 theta2 theta3
current      .81    .67    .53    .49
```

Because the policy is role-conditioned, record direction:

```text
theta_A as pursuer vs theta_B as evader
theta_A as evader  vs theta_B as pursuer
```

Do not assume symmetry.

---

# 24. Common policy interface

Define a common strategic-policy protocol so all of these can act interchangeably:

```text
current trainable policy
frozen snapshot
scripted pursuer
scripted evader
historical snapshot
```

For example:

```python
class StrategyPolicy(Protocol):
    def act(
        self,
        observation,
        role,
        state,
        key,
    ) -> tuple[Array, Any]:
        ...
```

Output must be the same high-level command representation.

The environment should not care whether the command came from PPO or a script.

---

# 25. Training both roles with one learner

One optimization batch may contain experience collected with the learner in both roles.

Each transition therefore needs explicit metadata:

```text
role
opponent_id
opponent_category
episode_id
environment_id
```

The reward and advantage calculation must respect role semantics.

Do not merge transitions in a way that loses role identity.

Expose metrics separately:

```text
pursuer/win_rate
pursuer/tag_time
pursuer/reward

evader/survival_rate
evader/survival_time
evader/reward
```

as well as aggregate metrics.

---

# 26. Data-generation requirement

Do not lose sight of the research objective while doing this architecture refactor.

MotionForge's eventual research question is still whether competitive interaction produces useful locomotion experience.

Therefore every hierarchical rollout should be capable of recording:

```text
strategy observation
role
high-level command
motor observation
motor action
root state
joint state
contacts
tracking error
fall / near-fall state
opponent state
terrain state
league opponent identity
game outcome
```

The model refactor should not make trajectory collection harder.

Prefer defining one structured transition type.

---

# 27. Suggested package organization

Move gradually toward:

```text
motionforge/
    models/
        hierarchical.py
        strategy.py
        locomotion.py
        types.py

    envs/
        physics/
            two_g1.py
            terrain.py

        tasks/
            tag.py

    policies/
        scripted/
            pursuer.py
            evader.py

        learned/
            hierarchical.py

    rollout/
        runner.py
        trajectory.py

    league/
        manager.py
        snapshot.py
        sampling.py
        metrics.py

    training/
        strategy_ppo.py
        locomotion.py
        hierarchical.py

    data/
        transition.py
        recorder.py

    evaluation/
        league.py
        tag.py
        locomotion.py
```

Do not perform a huge rename-only refactor before behavioral tests are in place.

Migration should be incremental.

---

# 28. Config structure

Add structured configuration for the architecture.

Something along the lines of:

```yaml
model:
  strategy:
    hidden_sizes: [128, 128]
    action_dim: 3
    role_conditioning: true

  locomotion:
    initialize_from: <accepted P2 checkpoint>
    trainable: true

  hierarchy:
    stop_gradient_command: true
    strategy_hz: 10
    locomotion_hz: 50

league:
  enabled: true

  sampling:
    recent: 0.55
    historical: 0.25
    best: 0.10
    scripted: 0.10

  snapshot_interval_updates: ...
  recent_capacity: ...
  historical_capacity: ...

roles:
  pursuer_probability: 0.5
```

Exact Hydra organization may follow existing MotionForge conventions.

---

# 29. Required tests before large training

Do not launch a long training run until these pass.

## Hierarchy tests

- strategy output has shape `(3,)`
- motor output has shape `(29,)`
- batched variants have correct leading dimensions
- all outputs finite
- command scaling correct
- temporal rates remain 10/50/500 Hz

## Role tests

Given identical physical state:

```text
role=pursuer
role=evader
```

must produce distinguishable network input.

Role must survive batching and JIT.

## Gradient tests

Motor-only loss:

```text
strategy gradient = zero
motor gradient != zero
```

Strategy loss:

```text
strategy gradient != zero
```

Verify actual parameter-tree gradients, not merely code-path assumptions.

## P2 parity

Before motor retraining:

```text
new locomotion module ~= old P2 locomotion controller
```

for deterministic observations.

## League tests

- deterministic sampling for fixed PRNG seed
- sampling frequencies approximately match configured probabilities
- frozen snapshot params never change
- snapshot fingerprints stable
- capacity eviction behavior deterministic
- scripted policy sampling works
- invalid probability config rejected

## Role/opponent tests

For one sampled matchup:

```text
learner=pursuer -> opponent=evader
learner=evader  -> opponent=pursuer
```

always.

## PPO smoke

Run a short GPU/JIT training smoke test with:

```text
role-conditioned learner
frozen historical opponent
```

Verify:

- finite losses
- learner parameters change
- opponent parameters do not
- both roles occur
- checkpoint saves/restores
- league metadata appears in logs

---

# 30. Migration sequence

Implement this in stages.

## Stage A — architecture skeleton

Create:

```text
StrategyEncoder
StrategyHead
LocomotionEncoder
MotorHead
HierarchicalPolicy
```

No league yet.

Load existing locomotion weights.

Verify P2 action parity.

## Stage B — role-conditioned strategic policy

Replace separate learned pursuer/evader architecture with one policy conditioned on role.

Initially train against existing scripted opponents.

Verify:

```text
same model can learn/use both roles
```

Do not introduce historical league complexity until this works.

## Stage C — trainable locomotion hierarchy

Enable motor optimization with its locomotion objective.

Add stop-gradient at the strategy-to-motor boundary.

Verify:

```text
strategy learns game objective
motor learns command execution
```

independently.

## Stage D — rollout abstraction

Build a rollout runner that takes:

```text
learner policy
learner role
opponent policy
opponent role
environment
```

and produces structured trajectories.

## Stage E — minimal league

Add:

```text
recent snapshots
scripted opponents
```

first.

Validate snapshot/self-play behavior.

## Stage F — full league categories

Add:

```text
historical
best
```

and payoff tracking.

Start with configurable:

```text
55% recent
25% historical
10% best
10% scripted
```

## Stage G — longer self-play

Only after all previous smoke tests pass:

```text
role-conditioned learner
+
trainable motor hierarchy
+
league-based historical self-play
```

---

# 31. Important non-goals for this refactor

Do **not** introduce yet:

- learned motor latent instead of `[vx, vy, yaw_rate]`;
- recurrent strategy networks unless existing observations prove insufficient;
- centralized critic;
- transformer architecture;
- simultaneous optimization of both active player copies;
- learned terrain generation;
- vision;
- population-based hyperparameter optimization;
- multiple humanoid morphologies;
- PARC integration;
- sim-to-real;
- end-to-end TAG reward directly training the motor controller.

Those can be explored after the basic hierarchical league architecture works.

---

# 32. Research interpretation

Keep the scientific distinction clear in documentation.

Old architecture:

```text
TAG learns strategy
motor capability fixed
```

New architecture:

```text
TAG strategy generates increasingly difficult behavioral demands
                       |
                       v
      trainable locomotion module encounters those demands
                       |
                       v
        locomotion objective improves execution
```

This lets us test whether competitive interaction changes the experience distribution enough to produce improved locomotor robustness.

The hierarchy is therefore not merely:

```text
network A calls network B
```

It is:

```text
strategic learner
       |
       v
behavioral command bottleneck
       |
       v
motor learner
```

with separate optimization objectives.

---

# 33. Success criteria for the refactor

Do not consider the architecture finished merely because code compiles.

Minimum completion criteria:

1. One parameterized hierarchical model contains both strategic and locomotion modules.
2. Strategy and locomotion use separate encoders.
3. Strategy outputs `[vx, vy, yaw_rate]`.
4. Stop-gradient exists between the strategic output and motor optimization path.
5. Locomotion parameters can actually update.
6. Strategy parameters can actually update.
7. One strategic model handles both pursuer and evader roles.
8. Training samples both roles.
9. A frozen opponent can be drawn from the league.
10. Recent/historical/best/scripted opponent categories are supported.
11. Opponent weights cannot change during a learner PPO segment.
12. Existing P2 locomotion behavior is reproduced before motor adaptation.
13. Existing deterministic tag evaluation remains available as a baseline.
14. All experiment manifests record architecture version, league state, opponent identity, role distribution, and parameter/checkpoint provenance.

---

# Final target architecture

```text
                        ROLE
                         |
                         v
                +----------------+
game state ---->| StrategyEncoder|
                +-------+--------+
                        v
                 StrategyHead
                        |
                  [vx,vy,wz]
                        |
                   stop-grad
                        v
                +------------------+
proprio ------->|LocomotionEncoder |
contacts ------>|                  |
gait ---------->|                  |
                +--------+---------+
                         v
                      MotorHead
                         |
                         v
                    29 actions
                         |
                         v
                      Physics
                     /       \
                    /         \
            TAG objective   motor objective
                 |               |
                 v               v
           strategy opt       motor opt


                       League
           +--------+-------+--------+
         recent historical best   scripted
           +--------+-------+--------+
                    v
             frozen opponent
                    v
                RolloutRunner
```

## Final implementation notes

- Do not take the word `head` too literally. The strategic and locomotion sides should each have their own encoder.
- The modules communicate through the high-level command bottleneck.
- The stop-gradient separates optimization, not information flow.
- The league is attached by ``LeagueEnv`` above the low-level ``WarpEnv``;
  policy execution and learner updates remain in the rollout/training layer.
- The initial league mix (`55% recent / 25% historical / 10% best / 10% scripted`) is provisional and must be configurable.
- The opponent is frozen during a learner PPO collection/update segment.
- The same role-conditioned learner can act as pursuer or evader.
- Keep the current sequential P7 architecture available as a reproducible baseline.
