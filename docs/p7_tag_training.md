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

Evidence: `logs/p7/tag_pursuer_environment_a.json` and
`logs/p7/tag_pursuer_networks_a.json`. The environment audit passed on the GPU
backend, including nine-value clock-free observations, exact action scaling,
five-update command holding, finite integrated physics, independent reward
fixtures, and timeout truncation semantics. The network audit initialized and
executed finite batched inference with the expected shapes and bounded sampled
actions. The actor has 18,566 parameters and the separate critic has 17,921.
