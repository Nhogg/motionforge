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

## Complete scripted tag rollout

The existing renderer now has an opt-in active-pursuer mode, so isolated evader
diagnostics and complete games share the same environment, locomotion bridge,
checkpoint restoration, camera, and artifact format. Agent 0 runs the scripted
pursuer, agent 1 runs the boundary-aware scripted evader, and both use the
accepted P2 parent controller. Red and blue model coloring identify pursuer and
evader respectively.

The seed-0 game ended in a valid tag at 4.90 seconds. Both robots remained
upright and inside the arena; minimum root height was 0.770 m and the minimum
planar root distance was 0.500 m. The renderer produced 148 frames and a JSON
sidecar containing configuration, final commands, termination flags, and
checkpoint provenance. Together with the earlier timeout rollout, this records
both primary complete episode outcomes without introducing another simulation
path.

Evidence:

- `logs/p4/videos/tag_scripted_complete_seed0.mp4`
- `logs/p4/videos/tag_scripted_complete_seed0.json`

## Rapid command-change evaluation

`scripts/evaluation/evaluate_g1_rapid_commands.py` evaluates the accepted P2
checkpoint independently of tag policy behavior. Five deterministic schedules
alternate every two seconds: forward/reverse, lateral left/right, yaw
left/right, diagonal reversal, and forward/brake. Four seeds produce 20
ten-second rollouts. Random pushes are disabled and physical fall termination
ignores incidental prohibited leg contacts, isolating command-transition
robustness.

Each rollout records physical survival, finite state, minimum root height,
maximum tilt, linear and yaw tracking RMSE, and recovery after every transition.
Recovery requires linear error at most 0.25 m/s and yaw error at most 0.30 rad/s
for 0.2 consecutive seconds. This is deliberately stricter than survival.

Evidence: `logs/p4/g1_rapid_commands_a.json`. All 20 rollouts remained finite
and survived their full 10 seconds. Lateral and diagonal reversals recovered on
100% of measured transitions. Forward/reverse recovered on 87.5%; yaw switching
and braking/restart each recovered on 93.75%. Maximum observed tilt remained
below 10.4 degrees. Physical survival is adequate for conservative scripted
rollouts, but the P4 rapid-change gate remains open: the repaired checkpoint
must recover on every transition in this fixed suite.

## Rapid-transition repair curriculum

`G1StandingJoystick` now exposes an opt-in rapid-command curriculum. When
enabled, each structured command is held for a uniformly sampled interval,
then replaced by another structured command or by its exact reversal. The
default interval is 1--3 seconds and the default reversal probability is 0.5.
Existing P2 training and evaluation behavior is unchanged unless
`rapid_command_transitions=true` is supplied.

The curriculum owns a separate transition counter rather than changing the
upstream Playground command sampler. On a transition it updates the command in
environment state and in both actor and privileged observation streams. It
also restarts Playground's longer command-resampling counter, preventing the
two schedules from issuing commands independently. Environment physics,
rewards, PPO, controller inference, and the rapid-command evaluator remain
separate modules.

The initial implementation applied rapid transitions to every environment.
The 40M continuation remained physically stable but regressed from the parent
checkpoint's 95% aggregate transition recovery to 77.5%. Intermediate
checkpoints show a consistent forgetting trend: 93.75% at 5M and 90% at 10M.
The final scenario rates were 75% forward/reverse, 68.75% lateral, 75% yaw,
68.75% diagonal, and 100% braking. This run is rejected and must not replace
the accepted P2 controller.

The revised curriculum samples rapid transitions for 25% of reset episodes;
the remaining 75% retain the original steady-command schedule as rehearsal.
This fraction is exposed as `rapid_command_episode_probability`. The repair
also reduces the learning rate to `1e-5` and limits the next candidate to 5M
steps before evaluation. Further training is conditional on fixed-suite
improvement, preventing another unchecked 40M continuation.

`scripts/tests/test_g1_command_transitions.py` fixes the interval at 0.04 s and
the reversal probability at one. It verifies the exact hold/reverse sequence,
both observation updates, counter reset, finite state, and Warp execution.
Evidence: `logs/p4/g1_command_transitions_mixed_a.json`; all checks passed,
including forced active and inactive episode paths.

The revised restored-policy PPO smoke ran the mixed curriculum for 102,400
steps from the accepted 40M structured checkpoint at a learning rate of
`1e-5`. It completed on the GPU, kept all metrics finite, recorded progress,
and wrote a checkpoint. This proves the training path but is not a candidate
policy. Evidence: `logs/p4/training/g1_rapid_transition_mixed_smoke/`.

The first mixed candidate continued for 5,079,040 steps at `1e-5`. It retained
20/20 physical survival and recovered on 78 of 80 transitions (97.5%),
improving on the parent checkpoint's 76/80 result. Forward/reverse, yaw, and
braking were perfect. Lateral seed 3 missed the final transition and diagonal
seed 0 missed the third transition, leaving both scenarios at 93.75%. The
candidate is the best measured checkpoint but does not yet pass the unanimous
recovery gate. Evidence:
`logs/p4/g1_rapid_commands_mixed_5m_seed0.json`.

Because the mixed distribution improved rather than degraded recovery, one
guarded continuation is authorized from this candidate: 2.5M additional steps
at `5e-6`, followed immediately by the unchanged evaluator. No longer run is
authorized without another measured improvement.

That guarded continuation was rejected. Although all 20 rollouts again
survived, recovery declined to 73 of 80 transitions (91.25%): 93.75%
forward/reverse, 93.75% lateral, 87.5% yaw, 87.5% diagonal, and 93.75%
braking. This demonstrates that additional PPO updates move the small number
of failures between scenarios rather than monotonically removing them. The 5M
mixed checkpoint remains the best candidate; the continuation must not be used.
Evidence: `logs/p4/g1_rapid_commands_mixed_plus2p5m_seed0.json`.

Further hyperparameter changes must not be selected against seeds 0--3. The
next measurement uses reset seeds 4--19 as a held-out confirmation set. This
separates general transition robustness from repeated optimization against an
80-transition development suite. Physical survival remains a hard requirement;
recovery results will be reported as a rate with scenario breakdown before the
P4 gate is reconsidered.

The held-out comparison rejects the mixed 5M candidate. Both controllers
completed all 80 held-out rollouts with finite state and no falls. The accepted
P2 parent recovered on 292 of 320 transitions (91.25%); the mixed candidate
recovered on 289 of 320 (90.31%). The candidate improved forward/reverse and
braking recovery but regressed yaw and diagonal recovery. Its small reductions
in aggregate tracking RMSE and maximum tilt do not outweigh the lower recovery
count. The P2 parent therefore remains the active low-level controller.

Across the fixed development and held-out suites, the parent survived all 100
ten-second rollouts under abrupt two-second command changes. This satisfies the
P4 requirement that the low-level controller survive rapid direction changes.
The stricter `all_transitions_recovered` evaluator check intentionally remains
false: survival acceptance does not imply perfect command tracking. No
rapid-transition fine-tuned checkpoint is promoted, and the measured recovery
limitation remains part of the P4 architecture record.

Held-out evidence:

- `logs/p4/g1_rapid_commands_parent_heldout_16.json`
- `logs/p4/g1_rapid_commands_mixed_5m_heldout_16.json`

The repair experiments are complete. Their curriculum and checkpoints remain
available as reproducible negative results, but no further fine-tuning is
planned for P4. Literal 100% recovery on a finite seed suite is not treated as
a safety claim: it depends on chosen error thresholds and encourages tuning to
the evaluator. Future controller work should use transition-aware rewards,
held-out schedules, and statistical recovery targets if tag behavior exposes
an operational need. The evaluator retains its strict
`all_transitions_recovered` diagnostic so imperfect tracking is never hidden.

## Integrated rollout audit

`scripts/evaluation/evaluate_tag_scripted_rollouts.py` is the final P4
integration gate. It runs the active scripted pursuer and boundary-aware
evader through the accepted P2 controller and the complete two-agent
MJX-Warp environment. Each run records its explicit seed and configuration in
JSON. Rendering remains separate so the quantitative audit has no camera or
encoding dependency.

The audit checks deterministic reset by resetting twice with an identical JAX
key and requiring bitwise-equal generalized positions and velocities. During
every control update it checks finite policy commands, normalized actions,
joint targets, generalized state, and reconstructed observations. Relative and
arena-center observation norms are independently recomputed from the same
pelvis body positions used by the observation module. At termination it checks
that exactly one terminal cause is active and that every reported tag has a
positive inter-agent contact count and a nonpositive signed contact distance.

The first audit draft compared pelvis-body observation coordinates against the
free-joint origins used for arena bounds. Those points separate slightly when
a robot tilts, so the check failed despite valid local frames. The invariant
was corrected to compare like-for-like pelvis-body positions; the environment,
controller, and policy were not changed to satisfy the audit.

Evidence: `logs/p4/tag_scripted_rollouts_a.json`. All eight 20-second seeded
rollouts passed on the GPU backend and ended in tags after 4.72--4.92 seconds.
There were no falls, out-of-bounds terminations, timeouts, nonfinite states, or
ambiguous terminal causes. Reset determinism, observation consistency,
controller-output finiteness, and tag-contact consistency passed for every
seed. This closes the final P4 pre-training integration gate.
