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

The JSONL format is an inspectable smoke-test artifact rather than the final
large-dataset storage decision. Later P5 fields will extend the same versioned
record boundary before a compact batch format is selected from measured data
volume.
