# P1 — MuJoCo stack setup

## Reproducibility conventions

- Python command-line interfaces use Tyro.
- Each automated check exposes a seed and configuration.
- Machine-readable results are stored under `logs/p1/`.
- Duplicate seeded runs are compared byte-for-byte where applicable.

## Standard MuJoCo

Python 3.11.15 and MuJoCo 3.12.0 were validated with a falling-sphere contact
test. Two seed-0 runs were byte-identical:

- `logs/p1/mujoco_smoke_a.json`
- `logs/p1/mujoco_smoke_b.json`

## MJX-JAX

The same model was run using MJX 3.12.0 and JAX 0.10.2. The validated test
used the CPU backend and produced byte-identical results:

- `logs/p1/mjx_test_a`
- `logs/p1/mjx_test_b`

The two-second rollout accumulated `1.8596649169921875e-05` seconds of float32
timing error, within the configured `0.0002`-second tolerance.

## MJX-Warp on Mayo

MJX-Warp was validated on `cuda:0`, an NVIDIA GeForce RTX 5090. The environment
used Warp 1.16.0 with its CUDA 12.9 runtime and NVIDIA driver 595.84. Both runs
passed and were byte-identical:

- `logs/p1/mjx_warp_a.json`
- `logs/p1/mjx_warp_b.json`

## Humanoid locomotion reproduction

The `dm_control` CMU humanoid example was first validated for deterministic
construction, stepping, observations, and interactive rendering. Random joint
actions caused the expected fall and were not counted as locomotion.

Actual locomotion was reproduced from MuJoCo Playground revision
`8a4b4642d8eba8a80ac99ed125cb62c16e1457ad` using its bundled G1 ONNX policy
and Menagerie revision `1b86ece576591213e2b666ebf59508454200ca97`.
The native MuJoCo 3.6.0 rollout used a fixed command of `[0.5, 0.0, 0.0]` for
10 seconds. The controller remained upright, executed 500 control updates and
5000 physics steps, and moved 5.812 metres forward. Duplicate results were
byte-identical:

- `logs/p1/g1_policy_a.json`
- `logs/p1/g1_policy_b.json`

The gait was visually verified in the offscreen EGL rendering
`logs/p1/g1_policy.mp4`. Offscreen rendering avoided the severe latency seen
with an interactive viewer over X11 forwarding.

## DeepMind multi-agent soccer

`dm_control` 1.0.45 was validated with two home and two away CMU humanoids.
Protobuf was constrained to `<7` because protobuf 7 removed a descriptor API
still used by the mocap loader.

Each player receives a separate 56-dimensional action, observation dictionary,
and reward, while all four share one MuJoCo physics state. Model bodies are
namespaced under `home0/`, `home1/`, `away0/`, and `away1/`. The compiled model
contained 137 bodies, 229 joints, 259 generalized positions, 254 generalized
velocities, and 224 actuators (`4 × 56`). Two 100-step seed-0 rollouts passed
and were byte-identical:

- `logs/p1/soccer_a.json`
- `logs/p1/soccer_b.json`

Interactive rendering confirmed four visually distinct players and a shared
field and ball. Random actions only produced flailing; no soccer policy was
trained, keeping policy training outside this P1 infrastructure check.

## Two-humanoid contact stability

A seeded 1-vs-1 `dm_control` soccer model was used strictly as a shared-physics
contact fixture. The two humanoid roots began 0.8 metres apart and were given
opposing 1.5 m/s velocities, for a 3.0 m/s closing speed. No TAG roles, rewards,
observations, or locomotion policies were added.

The first inter-humanoid contact occurred at 0.055 seconds. Contact was present
for 67 of 150 physics steps, with 147 contact instances and at most six
simultaneous player contacts. Peak measured contact force was 923.20 N, maximum
penetration was 20.74 mm, and peak absolute generalized acceleration was
4354.26. All states, accelerations, and forces remained finite throughout the
0.75-second test. Duplicate results were byte-identical:

- `logs/p1/humanoid_contact_a.json`
- `logs/p1/humanoid_contact_b.json`

The collision was visually confirmed using the slow-motion offscreen EGL
rendering `logs/p1/humanoid_contact.mp4`.

## Uneven-terrain contact stability

A deterministic 9-by-9 MuJoCo heightfield was tested using three spheres
constrained to move vertically at different terrain positions. This isolated
heightfield collision behavior from locomotion and controller behavior. The
seed-0 experiment ran for 2000 steps at a 0.002-second timestep, for four
seconds of simulated time.

All three spheres contacted the terrain and settled at distinct heights of
0.1605, 0.2406, and 0.2496 metres. Their final absolute vertical speeds were
below `8e-13` m/s. Contact persisted across 1833 steps; all states and contact
forces remained finite. Peak contact force was 78.79 N, maximum penetration
was 25.61 mm, and peak absolute generalized acceleration was 388.23. Duplicate
results were byte-identical:

- `logs/p1/heightfield_contact_a.json`
- `logs/p1/heightfield_contact_b.json`

This completes the P1 MuJoCo stack checks. Humanoid selection and controller
evaluation remain scoped to P2.

The heightfield fixture also supports visual inspection with its Tyro `--view`
flag. This renders an MP4 using MuJoCo's offscreen renderer; `--video-output`
sets its destination. Rendering does not change the default headless
experiment path.
