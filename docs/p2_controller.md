# P2 — Humanoid and locomotion prior

The evolving training-system architecture is recorded separately in
`docs/p2_training_architecture.md`.

## Morphology decision

The project will use the Unitree G1 as its single humanoid morphology.

This choice is based on the P1 reproduction rather than adding a new model:

- MuJoCo Playground provides a working G1 model and pretrained ONNX locomotion
  policy.
- The policy already accepts a three-dimensional command containing desired
  forward velocity, lateral velocity, and yaw rate.
- A deterministic 10-second forward-command rollout remained upright and
  travelled 5.812 metres.
- The model, policy artifact, Playground revision, and Menagerie revision are
  pinned in the P1 record.

The CMU humanoid remains useful as the P1 multi-body contact fixture, but it is
not the selected project morphology. Its tested soccer configuration did not
include a locomotion policy, whereas the G1 reproduction already supplies a
usable locomotion prior. Maintaining two morphologies would also violate the
minimum-scope constraint.

## Existing controller

The initial locomotion prior is MuJoCo Playground's bundled G1 ONNX policy,
identified by SHA-256
`db2eb258494c1297c43d2b9ffa94cdbde97654c2a44cbab0b40fd4b990752a5b` at
Playground revision `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`.

The current reproduction runs native MuJoCo at 0.002 seconds per physics step
and updates the controller every 0.02 seconds. It demonstrates that the model
and controller can perform basic commanded forward locomotion. Forward,
backward, turning, stopping, and recovery behavior must still be evaluated
explicitly before the controller interface is finalized.

## Movement reproduction

### Backward walking

A constant command of `[-0.5, 0.0, 0.0]` was reproduced for 10 seconds. The G1
remained upright with a minimum root height of 0.697 metres and moved 2.165
metres backward. It also accumulated 0.900 metres of lateral drift, which will
be retained as a controller-performance limitation rather than hidden by the
binary stability check. The two seed-0 outputs were byte-identical:

- `logs/p2/g1_backward_a.json`
- `logs/p2/g1_backward_b.json`

### Turning investigation

A preliminary pure-yaw command produced the expected positive rotation but
also unexpected horizontal translation. Diagnostics showed that this was
persistent pelvis-frame motion rather than a startup transient or coordinate
measurement artifact. With `[vx, vy, wz] = [0.0, 0.0, 0.5]`, mean local linear
velocity was `[0.230, -0.091]` m/s and mean yaw rate was 0.419 rad/s.

A forward-command sweep at fixed `wz = 0.5` demonstrated coupling between
linear and angular tracking. Commands `vx = [-0.30, -0.15, 0.0, 0.15, 0.30]`
produced mean local forward velocities of approximately
`[-0.134, -0.016, 0.230, 0.398, 0.495]` m/s and mean yaw rates of
`[0.526, 0.524, 0.419, 0.230, 0.207]` rad/s. A small backward command happened
to yield a near-stationary turn, but no compensation will be hidden in the
controller wrapper.

The upstream policy samples nonzero forward, lateral, and yaw commands jointly
and optimizes both linear and angular tracking rewards. The observed behavior
is therefore recorded as command coupling in this policy. Attributing the
coupling specifically to training-distribution coverage remains an inference;
it would require checkpoint comparison or retraining to establish causally.

Both uncompensated turn directions remained upright and were reproducible.
The left command `wz = 0.5` produced a mean yaw rate of 0.419 rad/s and mean
local linear velocity `[0.230, -0.091]` m/s. The mirrored right command
`wz = -0.5` produced -0.376 rad/s and `[0.162, 0.052]` m/s. Thus lateral drift
changed sign with turn direction while forward creep remained positive. The
right-turn seed-0 outputs were byte-identical:

- `logs/p2/g1_turn_right_a.json`
- `logs/p2/g1_turn_right_b.json`

### Stopping investigation

After a five-second forward command followed by five seconds of zero command,
the G1 continued along 1.265 metres of horizontal path. A separate rollout
initialized with a zero command exhibited persistent forward drift: mean local
forward velocity was 0.133 m/s in the first quarter and 0.134 m/s in the last
quarter. This rules out braking momentum as the sole explanation.

The upstream G1 environment advances its gait phase at zero command and
contains a disabled, explicitly documented option to freeze the phase to make
the policy stand still. The reproduction exposes this behavior as the opt-in
`--freeze-phase-at-zero` diagnostic while preserving the upstream default for
controlled comparison.

The phase-freeze diagnostic did not produce a viable stance. From the initial
pose, the G1 fell and reached a minimum root height of -0.752 metres. Freezing
sets both leg phases to `[pi, pi]`, whereas the enabled training behavior keeps
the two phases separated by pi radians and advances them continuously. Because
the freeze logic is commented out in the pinned training environment, the
same-phase input was not part of that enabled behavior and is unsafe for this
exported policy. Phase freezing is rejected as the stopping solution.

The bundled ONNX policy is therefore confirmed to locomote and turn, but it
does not provide a satisfactory zero-command stop by itself. A stopping mode
must be validated before this P2 checkpoint can be marked complete.

A second diagnostic bypassed the policy at zero command and held the
`knees_bent` keyframe actuator targets. The keyframe controls exactly match its
29 joint positions, but the G1 still fell, reaching a minimum root height of
-0.779 metres. The keyframe is an initialization pose rather than a
self-balancing static controller, so fixed pose hold is also rejected.

## MotionForge controller boundary

After structured-command training and acceptance evaluation, MotionForge uses
checkpoint `g1_structured_finetune_40m_seed0/000040632320` as the current
locomotion prior. The public high-level command is a three-element vector:

```text
[desired pelvis-frame forward velocity (m/s),
 desired pelvis-frame lateral velocity (m/s),
 desired yaw rate (rad/s)]
```

`G1VelocityCommand` supplies names and units for configuration boundaries and
serializes to the policy ordering `[vx, vy, yaw_rate]`. No clipping,
compensation, smoothing, hidden nonzero command, or gait-phase override is
applied. The caller owns command scheduling.

`apply_velocity_command` writes that array to the environment's command state
and to indices 9--11 of both policy and privileged observations. Observation
construction remains owned by the environment.

`G1LocomotionController` owns restored PPO parameters and deterministic policy
inference. Given an environment-produced observation and a JAX random key, one
control update returns:

- `normalized_action`: 29 policy actions consumed by `environment.step`;
- `joint_position_targets`: the same actions mapped through Playground's
  `default_pose + normalized_action * action_scale` actuator contract.

The controller does not reset or advance physics, mutate a command, choose a
role, calculate game observations, or own logging. The environment remains the
only component that advances MuJoCo/MJX. This makes one future two-agent step
explicit:

```text
high-level command [vx, vy, yaw_rate]
  -> environment-produced policy observation
  -> checkpoint-backed G1 controller
  -> 29 normalized actions / joint-position targets
  -> environment physics step
```

The interface implementation lives in `motionforge/controllers/g1.py`. Its
smoke test verifies command round-tripping, action and target shapes, exact
target conversion, deterministic inference, a successful environment step,
finite state, and GPU execution. P3 composition is intentionally not included
in this P2 interface change.

The seed-0 interface smoke passed on `cuda:0` with JAX 0.10.2 and MuJoCo
3.12.0. Command `[0.3, -0.2, 0.5]` round-tripped through both policy
observations, deterministic inference emitted 29 finite normalized actions,
and their 29 joint-position targets exactly matched the environment's
actuator conversion. Applying the normalized action advanced one finite MJX
physics step. This validates both the low-level controller contract and the
public `[vx, vy, yaw_rate]` high-level interface, completing P2.

- `logs/p2/g1_controller_interface_a.json`

The pinned Playground revision contained no alternative bundled policy that
could solve the original zero-command weakness. MotionForge addressed it by
training the accepted checkpoint with an explicit standing mode and structured
command distribution, rather than hiding it behind a command offset or an
unstable supervisory controller.
