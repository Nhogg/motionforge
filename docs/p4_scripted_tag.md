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

## Scripted evader

The evader uses the same local relative-position input and emits a command in
the opposite direction. Its translational command is proportional to the
negative relative displacement, and its yaw command turns toward that escape
direction. Independent gains and limits allow later rollout diagnostics to tune
the evader without changing pursuit behavior.

The initial evader was deliberately unaware of arena boundaries. Its first
rendered rollout remained upright but exited the arena at 7.22 s. The accepted
revision consumes the environment's local-frame vector toward arena center. At
a conservative radial safety margin it temporarily overrides escape behavior
and commands inward motion, keeping arena rules out of the locomotion adapter.

Evidence: `logs/p4/tag_scripted_evader_e.json`. All 12 deterministic checks
passed on the GPU backend, including directional signs, inward boundary
correction, JIT execution, finiteness, output shape, and saturation at each
configured command limit. The scripted pursuer regression test also remained
green.

## Two-agent locomotion bridge and evader render

`motionforge/controllers/g1_tag.py` reconstructs the accepted P2 actor's
103-element observation independently for each robot from the combined P3
physics state. It includes local pelvis velocity, gyro, projected gravity,
high-level command, joint position and velocity, previous action, and gait
phase. The adapter returns actor state only; privileged training observations
are not required for deterministic checkpoint inference. Policy restoration,
joint-target conversion, and physics stepping remain in their existing
controller and environment modules.

`scripts/evaluation/render_tag_scripted_evader.py` exercises this bridge with
the accepted structured-command checkpoint. Agent 0 receives a zero locomotion
command while agent 1 recomputes its scripted escape command from the live
relative observation every control update. The renderer follows the midpoint
of both agents and writes an MP4 plus a JSON sidecar containing configuration
and the terminal cause.

The first hard boundary override prevented OOB but issued an aggressive
full-speed reversal and caused a fall at 8.28 s. Reducing the evader limits to
0.5 m/s forward, 0.25 m/s lateral, and 0.6 rad/s yaw produced a stable result.
This failed intermediate rollout is retained as evidence of the low-level
controller's sensitivity to rapid command changes.

Evidence: `logs/p4/videos/tag_scripted_evader_boundary_d.mp4` and its JSON
sidecar. The seed-0 boundary-aware rollout completed 10 seconds with both agents
upright and in bounds, no tag contact, and timeout as its only terminal cause.
