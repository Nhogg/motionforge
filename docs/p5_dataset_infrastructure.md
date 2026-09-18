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

## Opponent-relative state

Schema version 6 records the canonical relative observation already computed
by `TwoG1TagEnvironment`. `opponent_relative_position_heading` and
`opponent_relative_velocity_heading` each have shape `(2, 2)`: one planar
vector per observer, rotated into that observer's yaw-aligned heading frame.
The logger validates these exogenous observation arrays instead of introducing
a second coordinate transform that could drift from policy inputs.

The smoke experiment checks finite values and full trajectory shapes. It also
verifies that both agents initially observe the configured separation, while
allowing their vector components to differ because each observation uses its
own heading frame.

## Terrain information

Schema version 7 records the P3/P4 flat-terrain assumption explicitly. During
setup, the logger resolves the shared `floor` geom and requires it to be a
horizontal MuJoCo plane. Each timestep records `terrain_kind=flat_plane`,
`terrain_height` with shape `(2,)`, and `terrain_normal_world` with shape
`(2, 3)`. Repeating the static sample per agent keeps the timestep schema
uniform and leaves a clear replacement point for future local height or normal
queries on non-flat terrain.

The smoke experiment verifies that the recorded heights match the compiled
floor, all normals are finite unit vectors, and every row declares the same
terrain kind. No terrain curriculum or non-flat model is introduced in P5.

## Reward and game outcome

Schema version 8 adds a validated per-agent `game_reward` vector and the
canonical sticky termination fields: tag, per-agent fall, per-agent
out-of-bounds, timeout, and aggregate done. Rewards are supplied by the caller
because P5 does not define the learned-game objective planned for P7. The smoke
experiment therefore records neutral zero rewards rather than introducing a
premature reward function.

`game_winner_index` is derived from the fixed role assignment and terminal
cause. A tag awards the outcome to the pursuer, a timeout to the evader, and a
single-agent fall or boundary exit to the other agent. It is `-1` while the
episode is active or when a simultaneous failure has no unique winner. Direct
fixtures verify tag and timeout classification in addition to the per-timestep
shape, finiteness, and ongoing-outcome checks.

## Tracking error

Schema version 9 records `command_velocity_tracking_error` with shape `(2, 3)`
using the explicit convention `command - measurement`. The first two measured
components come from each pelvis's local linear-velocity sensor and the third
comes from the pelvis gyro yaw rate. These are the same coordinate conventions
used by the locomotion controller, so heading-frame commands are never compared
against world-frame root velocity.

The smoke experiment validates finite values and the full trajectory shape.
The vector form is retained instead of reducing immediately to a norm so later
analysis can distinguish forward, lateral, and yaw tracking failures.

## Fall and near-fall indicators

Schema version 10 records the canonical instantaneous environment fall flag,
root height, and torso up alignment for both agents. It also records a
configurable `stability_near_fall` flag. A state is near-fall when it has not
already been classified as fallen and either root height is below 0.60 m or up
alignment is below 0.80 by default. Both thresholds are exposed in experiment
configuration and written into the summary.

Keeping fall and near-fall mutually exclusive distinguishes warning states
from failures while retaining the continuous diagnostics needed to retune the
threshold later. The smoke experiment validates shapes, finite diagnostics,
exclusivity, and a fixture containing one near-fall and one true fall.

## Linear acceleration

Schema version 11 adds `world_linear_acceleration`, derived offline from the
recorded world-frame root velocity with a first-order difference divided by the
control timestep. Derivation lives in `motionforge/logging/tag_derivatives.py`,
separate from online state extraction and simulation stepping.

The output retains the input trajectory length and shape `(T, 2, 3)`. Timestep
zero is filled with zeros but explicitly marked false by
`world_linear_acceleration_valid`, because no preceding sample exists. Every
later timestep is marked valid. The smoke experiment checks finite values,
shape, validity semantics, and an exact synthetic constant-acceleration
fixture.

## Yaw acceleration

Schema version 12 adds `pelvis_yaw_acceleration`, derived from the z component
of the recorded pelvis-frame angular velocity. It uses the same first-order
difference and control timestep as linear acceleration, producing shape
`(T, 2)` with a parallel `pelvis_yaw_acceleration_valid` mask. Timestep zero is
zero-filled and invalid because no preceding yaw-rate sample exists.

The smoke experiment checks finite values, shape, validity semantics, and an
exact two-agent synthetic fixture with different yaw accelerations.

## Command derivatives

Schema version 13 adds `command_velocity_derivative`, the first-order time
derivative of the recorded high-level `[vx, vy, yaw_rate]` command. The output
has shape `(T, 2, 3)` and units `[m/s², m/s², rad/s²]`. As with the physical
derivatives, timestep zero is zero-filled and marked invalid by a parallel
`command_velocity_derivative_valid` mask.

The smoke experiment checks shape, finiteness, validity semantics, and a
synthetic command transition with exact expected component-wise rates.

The JSONL format is an inspectable smoke-test artifact rather than the final
large-dataset storage decision. Later P5 fields will extend the same versioned
record boundary before a compact batch format is selected from measured data
volume.
