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
