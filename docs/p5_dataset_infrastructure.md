# P5 logging and dataset infrastructure

P5 adds a versioned, per-control-timestep trajectory schema to the integrated
tag environment. Logging remains downstream of simulation and policy
inference: extractors read immutable environment state, while experiments own
host transfer and serialization. This keeps dataset generation optional and
prevents file I/O from entering JIT-compiled control loops.

## Root pose and orientation

`motionforge/logging/tag_trajectory.py` defines schema version 1 and the first
state extractor. A setup-time layout resolves each agent's floating-base
generalized-position address from the compiled two-G1 model. The JAX-compatible
extractor returns world-frame position with shape `(2, 3)` and normalized wxyz
orientation with shape `(2, 4)` directly from device state.

`scripts/tests/test_tag_root_pose_logging.py` exercises actual per-timestep
recording independently of learned policy inference. It accepts an explicit
seed and contact capacities, writes append-only JSONL trajectory rows, and
writes a separate machine-readable JSON summary. Every row contains the schema
version, episode seed, integer timestep, simulation time, and both root poses.
The initial state is timestep zero, followed by one row per completed control
update.

## Root linear and angular velocity

Schema version 2 adds root linear and angular velocity through a separate
setup-time sensor layout and JAX-compatible extractor. Linear velocity comes
from each model's `global_linvel_pelvis` sensor and is therefore recorded in
the world frame. Angular velocity comes from `gyro_pelvis` and is recorded in
the pelvis frame. The frame names are part of the JSONL field names rather
than implicit metadata: `world_linear_velocity` and
`pelvis_angular_velocity`, each with shape `(2, 3)`.

The logging smoke experiment validates both velocity arrays at every recorded
timestep for shape and finite values. Schema version 1 remains interpretable
as pose-only; version 2 is its additive root-state extension.

## Joint positions and velocities

Schema version 3 adds the raw actuated joint state for both robots. A dedicated
layout removes the seven floating-base qpos coordinates and six floating-base
qvel coordinates from each agent's namespaced model slices. The extractor
therefore produces `joint_position` and `joint_velocity` arrays with shape
`(2, 29)` in compiled actuator order. Values are recorded without subtracting
the controller's reference pose so downstream analyses retain the physical
simulator state.

The smoke experiment checks every joint sample for finite values and verifies
the complete `(timesteps, 2, 29)` shapes. Schema version 3 is an additive
extension of the pose and root-velocity records.

## Controller commands

Controller commands are exogenous inputs and cannot be recovered reliably from
the resulting physics state. Schema version 4 therefore adds a validated input
boundary that the rollout loop calls when it records each timestep. It stores
the high-level `command_velocity` with shape `(2, 3)` in `[vx, vy, yaw_rate]`
order and the applied `command_joint_position_target` with shape `(2, 29)`.

The logging smoke uses zero high-level velocity commands and the environment's
default joint targets, keeping this schema test independent of checkpoint
inference. It verifies finite command values, exact shapes, and equality
between the last logged joint targets and the controls present in simulator
state. Scripted and learned rollout generators will pass their live commands
through the same boundary.

## Foot contacts and contact forces

Schema version 5 adds per-agent left/right foot interaction with the shared
floor. A setup-time layout resolves the four namespaced foot geom IDs and the
floor geom ID. The Warp extractor filters active fixed-capacity contact slots,
requires nonpositive signed distance, and records `foot_contact` as a `(2, 2)`
boolean array in `[agent, left/right]` order.

For each active foot-floor contact, `foot_contact_normal_force` sums the raw
normal constraint force addressed by MuJoCo's contact record. The resulting
`(2, 2)` array is in simulator force units (Newtons). Tangential components
are not folded into this value; later slip-event derivation will combine the
contact mask with foot kinematics explicitly. The smoke experiment verifies
shapes, finite nonnegative forces, and the presence of at least one foot-floor
contact during the rollout.

The JSONL format is an inspectable smoke-test artifact rather than the final
large-dataset storage decision. Later P5 fields will extend the same versioned
record boundary before a compact batch format is selected from measured data
volume.
