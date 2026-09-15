# P4 scripted tag architecture

P4 validates the complete two-agent control path using deterministic scripted
game policies before any tag training. It builds on the integrated flat-ground
P3 environment and the accepted P2 G1 locomotion checkpoint.

## Module boundary

High-level tag behavior lives in `motionforge/policies`. A tag policy consumes
environment observations and emits a three-element G1 velocity command:
forward velocity, lateral velocity, and yaw rate. It does not step physics,
restore the locomotion checkpoint, or generate joint targets. Those concerns
remain in the environment and controller modules established during P2 and P3.

## Scripted pursuer

`motionforge/policies/tag_scripted.py` defines the first P4 policy. The pursuer
uses the opponent's planar position in its own heading frame. Proportional
forward and lateral commands point toward the opponent, while the relative
bearing supplies a proportional yaw-rate command. Each component is clipped to
an explicit limit compatible with the G1 controller. Translation becomes zero
inside a configurable tag radius, avoiding unnecessary high-speed contact.

The policy is stateless and uses only JAX array operations, allowing it to be
compiled and batched later without coupling it to the environment runtime.
This initial controller is intentionally simple: its purpose is to expose
integration and rapid-command problems before learned tag behavior is added.

Evidence: `logs/p4/tag_scripted_pursuer_a.json`. The deterministic command test
passed all ten checks on the GPU backend, covering target direction, close-range
translation stopping, output shape and finiteness, JIT execution, and command
limits.
