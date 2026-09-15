# P3 MVP environment architecture

This document records the P3 two-agent tag environment as it develops. P3 uses
the trained G1 locomotion controller from P2 and adds only the environment state
and rules needed for a reproducible tag MVP. Later-phase training, dataset, and
self-play infrastructure is intentionally excluded.

## Backend boundary

MJX-Warp is the runtime and training backend. Native MuJoCo remains a supporting
tool for MJCF compilation, deterministic state construction, focused model
checks, and video rendering. Features are not intended to have two independent
environment implementations. Shared role, observation, and rule logic should
operate on array state usable by the MJX-Warp runtime.

The current flow is:

```text
Menagerie G1 MJCF
       |
       v
native MuJoCo model composition and initialization
       |
       v
MJX-Warp runtime -> tag state and rules -> controller commands
       |
       +-----------------------------> native MuJoCo rendering
```

## Two-agent model composition

`motionforge/envs/two_g1.py` attaches two copies of Playground's pinned
`g1_mjx_feetonly.xml` to one flat arena. Names are isolated with stable prefixes:

- agent 0: `agent0/`
- agent 1: `agent1/`

Each `G1ModelLayout` records that agent's generalized-position slice,
generalized-velocity slice, and 29 actuator IDs. This prevents downstream code
from relying on duplicated hard-coded offsets. The compiled shared model has:

- 61 bodies;
- 72 generalized positions;
- 70 generalized velocities;
- 58 actuators.

`make_two_g1_data` initializes both agents from the accepted G1 `knees_bent`
keyframe and places them symmetrically about the origin along the world Y axis.
The requested separation is exposed as an experiment parameter.

Evidence:

- `logs/p3/two_g1_model_a.json`
- `logs/p3/two_g1_mjx_warp_a.json`
- `logs/p3/videos/two_g1_environment.mp4`

## Role assignment

`motionforge/envs/tag_roles.py` keeps game roles separate from physical model
layout. A `TagAgent` binds a stable agent index and model prefix to either the
`pursuer` or `evader` role. `assign_tag_roles` exposes the pursuer index so role
reversal is deterministic and testable without rebuilding the physics model.

Evidence:

- `logs/p3/tag_roles_agent0_pursuer.json`
- `logs/p3/tag_roles_agent1_pursuer.json`

## Relative observations

`motionforge/envs/tag_observations.py` extracts the other agent's planar
position and linear velocity relative to an observer. Both vectors are rotated
from the world frame into the observer's yaw-aligned heading frame. This makes
the observation independent of absolute arena heading while preserving the
direction and closing motion needed for tag.

Model addresses such as pelvis body IDs and velocity sensor slices are resolved
once into `TagObservationLayout`. Runtime extraction therefore remains array
based and does not perform name lookups.

The current observation intentionally contains only relative planar position
and velocity. Additional state should be added only when its corresponding P3
item requires it.

Evidence: `logs/p3/tag_relative_observations_a.json`.

## Tag contact detection

Tag interaction uses 64 explicit inter-agent collision pairs. Eight selected
collision geoms on each G1 form an 8 by 8 cross-agent pairing:

- left and right thighs;
- left and right shins;
- left and right feet;
- left and right hand collision geoms.

Floor contacts and contacts within one robot are excluded from tag detection.
`motionforge/envs/tag_contact.py` resolves geom ownership once into
`TagContactLayout`, then reports whether an inter-agent contact occurred, the
number of matching active contacts, and their minimum signed distance.

Native MuJoCo was used to validate model pair construction and contact
semantics. The integrated P3 environment test will exercise the detector through
the selected MJX-Warp runtime rather than maintaining a duplicate native runtime
path.

Evidence: `logs/p3/tag_contact_b.json`. The test produced 74 total model pairs:
10 inherited pairs plus 64 inter-agent pairs. At 2 m separation it found zero
tag contacts; in the overlapping fixture it found 28, with a minimum signed
distance of approximately -0.16 m.

## Fall detection

`motionforge/envs/tag_fall.py` classifies each agent independently using its
floating-base height and orientation. `FallDetectionLayout` resolves the two
root generalized-position addresses once, while `detect_falls` performs only
array operations suitable for the MJX-Warp runtime.

The initial thresholds are:

- minimum root height: 0.45 m;
- minimum torso-up alignment with world Z: 0.50, equivalent to a maximum tilt
  of 60 degrees for a normalized root quaternion.

An agent is classified as fallen when its root is below the height threshold,
its up alignment is below the orientation threshold, or its required root state
is non-finite. Diagnostics expose the per-agent fall flag, root height, up
alignment, and finite-state flag.

This classifier is deliberately stateless. It identifies an invalid posture at
one instant but does not yet terminate an episode. Temporal persistence belongs
in the later integrated environment, where a short consecutive-step counter can
avoid ending an episode because of a recoverable transient posture.

Evidence: `logs/p3/tag_fall_a.json`. Both initialized agents were upright at a
root height of approximately 0.755 m and an up alignment of 1.0. Lowering agent
0 to 0.44 m classified only agent 0 as fallen. Rotating agent 1 sideways to an
up alignment of 0.0 classified only agent 1 as fallen.

## Out-of-bounds detection

`motionforge/envs/tag_bounds.py` defines a logical square playing area centered
on the world origin. The initial arena half-extent is 4.0 m, so legal root
positions satisfy `abs(x) <= 4.0` and `abs(y) <= 4.0`. The flat physical floor
remains larger than the logical game boundary; leaving the playable area is an
episode rule rather than a collision with a wall.

`BoundsDetectionLayout` resolves both floating-base position addresses once.
The array-only runtime detector reports each agent's planar position, signed
maximum-axis boundary excess, finite-state validity, and OOB flag. A root
exactly on the boundary remains in bounds. Crossing either axis or producing a
non-finite planar position is OOB.

Evidence: `logs/p3/tag_bounds_a.json`. The deterministic fixtures cover both
agents and both planar axes, the inclusive boundary, isolation between agents,
and non-finite state handling.

## Deterministic reset sampling

`motionforge/envs/tag_reset.py` defines the array-level reset sampler used by
the future integrated MJX-Warp environment. It accepts explicit template qpos
and qvel arrays plus a JAX PRNG key, avoiding hidden global random state.

Each reset preserves the accepted knees-bent joint configuration, zeros all
generalized velocities, and places the agents symmetrically around the world
origin at a configurable separation. A seeded angle rotates the complete spawn
about world Z. Agent 0 faces along that angle toward agent 1; agent 1 faces in
the opposite direction toward agent 0. This samples rotationally distinct but
geometrically equivalent encounters without introducing a positional advantage
for either role.

`TagResetLayout` resolves root and joint position addresses once. The sampler
returns qpos, qvel, the sampled angle, planar positions, and headings, and uses
only JAX array operations so it can be compiled into the MJX-Warp reset path.

Evidence: `logs/p3/tag_reset_a.json`. The test requires exact equality for
repeated use of one seed, a changed spawn for a different seed, exact requested
separation, origin-centered positions, normalized root quaternions, mutually
facing headings, preserved joint poses, zero velocities, finite state, and
positions inside the current arena bounds.

## Episode timeout

`motionforge/envs/tag_timeout.py` defines episode duration in control steps from
an explicit duration and control timestep. The initial configuration is 20.0 s
at 0.02 s per control update, giving exactly 1,000 steps. Configuration rejects
durations that are not integer multiples of the control timestep so timeout
rounding cannot silently vary between experiments.

The timeout is false through step 999 and becomes true at step 1,000. Advancing
from step 999 therefore produces the terminal timeout state. Remaining steps
are clamped at zero after the boundary. Pure JAX functions separately support
observing the current counter and advancing it, allowing the integrated
MJX-Warp environment to define its update order explicitly.

Evidence: `logs/p3/tag_timeout_a.json`. The test covers initial, penultimate,
exact-boundary, post-boundary, and advance-to-boundary states under JIT.

## Flat terrain

P3 intentionally uses the simplest terrain required for tag: one infinite,
level MuJoCo plane named `floor`, centered at world Z = 0 with identity
orientation. Both G1 instances share this geom. There are no heightfield assets
or physical arena walls; the separate logical bounds detector defines the
playable square without changing contact dynamics near its edge.

This terrain was present from the first two-agent model composition. The final
P3 check makes that design constraint explicit and guards against accidentally
introducing terrain complexity before the flat-ground MVP is validated.

Evidence: `logs/p3/tag_flat_terrain_a.json`. The structural test verifies one
shared plane, no heightfields, level placement at world zero, equal initialized
root heights above the floor, and preservation of the configured simulator
timestep. Dynamic MJX-Warp stepping and floor contacts were previously verified
by `logs/p3/two_g1_mjx_warp_a.json`.

## Reproducibility conventions

P3 experiment scripts expose seeds and relevant physical configuration through
Hydra structured configs and write machine-readable JSON beneath `logs/p3`.
Hydra's timestamped output tree and duplicate job log are disabled by default
because each MotionForge experiment owns its explicit output path. CLI values
use Hydra's `field=value` override syntax. Model layout, role
assignment, observation extraction, contact rules, rendering, and tests remain
separate modules so later changes can be evaluated without replacing working
infrastructure.
