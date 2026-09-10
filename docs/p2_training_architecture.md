# P2 — Locomotion training architecture

## Objective

Train a Unitree G1 locomotion prior that preserves the working locomotion of
the bundled MuJoCo Playground policy while explicitly supporting a stable
zero-command mode. This remains a P2 prerequisite; the two-agent environment
does not begin until the locomotion interface passes its movement checks.

## Ownership boundary

MotionForge owns the experiment while a pinned MuJoCo Playground checkout
provides the simulator environment and PPO implementation.

### MotionForge owns

- the Tyro command-line interface and complete training configuration;
- the G1 environment subclass used for project-specific command sampling;
- experiment seeds, identifiers, manifests, and dependency revisions;
- training and evaluation output locations under `logs/p2/`;
- checkpoint selection and policy export;
- deterministic native-MuJoCo acceptance tests;
- the definition of forward, backward, lateral, turning, stopping, and
  recovery success.

Executable validation scripts are organized under `scripts/tests/`. Reusable
environment and training code lives in the installable `motionforge` package;
tests import it rather than manipulating `sys.path`.

### MuJoCo Playground owns

- the Unitree G1 model and flat-terrain MJCF;
- MJX/Warp physics integration;
- the base G1 observation and action definitions;
- the base reward implementation and termination rules;
- Brax PPO networks, rollout wrappers, and optimization code.

Playground source is not copied into MotionForge and is not modified in place.
It is consumed as an editable relative-path dependency pinned by its Git
revision. This allows MotionForge code to remain small while preserving the
exact upstream implementation used by each experiment.

## Environment specialization

The initial MotionForge environment subclasses Playground's flat-terrain G1
joystick environment. Its first and only behavioral override is command
sampling:

1. With a configured `standing_probability`, emit exactly `[0, 0, 0]`.
2. Otherwise sample forward, lateral, and yaw commands from the upstream
   configured ranges.

Standing is therefore an explicit training mode rather than a measure-zero
corner of the moving-command distribution. The probability is part of the
saved configuration and experiment manifest.

The initial version will not freeze gait phase, apply a hidden command offset,
or bypass the learned policy with fixed pose control. P2 diagnostics showed
that both gait-phase freezing and fixed keyframe control destabilize the
bundled policy.

## Training flow

```text
Tyro training configuration
          |
          v
MotionForge G1 environment registration
          |
          v
Pinned Playground G1 flat-terrain environment
          |
          v
Playground Brax PPO + MJX/Warp on GPU
          |
          v
logs/p2/training/<experiment>/
  manifest.json
  metrics.jsonl
  checkpoints/
          |
          v
MotionForge deterministic acceptance evaluator
          |
          v
Selected policy artifact + evaluation logs
```

## Reproducibility contract

Before training begins, each run records:

- MotionForge working-tree revision and status;
- Playground revision and status;
- Menagerie revision;
- Python, MuJoCo, MJX, Warp, JAX, and Brax versions;
- visible accelerator devices;
- seed and complete environment/PPO configuration;
- standing-command probability;
- output directory and checkpoint policy.

Training metrics are append-only machine-readable records. Checkpoints remain
inside the run directory and are never selected only by an informal visual
inspection.

## Validation stages

1. **Environment smoke:** sample resets and confirm both standing and moving
   commands occur at the configured rate on the GPU backend.
2. **Training smoke:** run a deliberately small PPO job to validate compilation,
   metrics, checkpointing, and restart behavior. It is not a policy-quality
   result.
3. **Full flat-ground training:** train from scratch with fixed seed/config and
   retain checkpoints.
4. **Checkpoint evaluation:** evaluate identical seeded command suites for
   forward, backward, lateral, turning, stopping, and push recovery.
5. **Selection:** choose a checkpoint by declared aggregate metrics and record
   its digest and provenance.
6. **Export:** export the selected policy for native MuJoCo only after the
   checkpoint passes the acceptance suite.

## Current artifact constraint

The bundled `g1_policy.onnx` is an inference artifact. It does not include the
PPO value network, optimizer state, observation-normalizer state in Brax
checkpoint form, or the original training checkpoint. MotionForge will
therefore train a new policy rather than claim to fine-tune that ONNX file.

## Environment setup record

The pinned sibling Playground checkout is installed into MotionForge as the
editable `playground` 0.2.0 dependency. The unified environment was validated
with MuJoCo 3.12.0, JAX 0.10.2, and Brax 0.14.2. JAX selected the GPU backend
and exposed the RTX 5090 as `cuda:0`.

The first MotionForge `G1StandingJoystick` reset also completed through
MJX-Warp on the GPU. It retained Playground's 103-element policy observation
and 36-element generalized-position state. The environment configuration
reported the intended standing probability of 0.30; its seeded reset sampled
a valid moving command `[0.0926, -0.1585, 0.1190]`.

The seeded command-distribution smoke test sampled 4096 commands on `cuda:0`.
It produced 1249 standing commands and 2847 moving commands, for an observed
standing probability of 0.30493 against the configured 0.30. All moving
commands remained within their configured ranges, and all command and reset
state values were finite. Two runs were byte-identical:

- `logs/p2/g1_command_sampling_a.json`
- `logs/p2/g1_command_sampling_b.json`

This completes the environment-sampling smoke stage. It does not establish
policy quality; the next stage is a deliberately small PPO training smoke test
covering compilation, metrics, and checkpoint creation.

## Training-stack compatibility

The first PPO smoke attempt stopped before training because Brax 0.14.2 calls
`jax.device_put_replicated`, an API removed by MotionForge's JAX 0.10.2. The
pinned Playground lock previously used JAX 0.6.2, but downgrading MotionForge
would replace the CUDA 13 stack already validated in P1.

Until a compatible Brax release is pinned, MotionForge supplies a narrow
compatibility adapter implementing only `device_put_replicated` using JAX's
documented `device_put`, `Mesh`, and `NamedSharding` replacement. The adapter
is installed before PPO starts and its activation is recorded in the training
manifest. Brax and Playground source remain unmodified.

The adapter allowed the first PPO smoke run to complete on `cuda:0`. The run
processed 101,120 steps (the first complete PPO batch beyond the requested
100,000), emitted two machine-readable progress records, kept all recorded
metrics finite, and created checkpoint `000000101120`. Total wall time was
42.7 seconds, including compilation and evaluation:

- `logs/p2/training/g1_standing_smoke_compat_a/manifest.json`
- `logs/p2/training/g1_standing_smoke_compat_a/metrics.jsonl`
- `logs/p2/training/g1_standing_smoke_compat_a/summary.json`
- `logs/p2/training/g1_standing_smoke_compat_a/checkpoints/000000101120/`

The final evaluation reward was -3.04 and average episode length was 27.9
control steps. These are deliberately not treated as policy-quality results:
the smoke run used only about 0.05 percent of Playground's 200-million-step G1
training budget. This checkpoint validates infrastructure only and will not be
selected as a locomotion prior.

A second seed-0 Warp smoke run also completed at 101,120 steps and created the
same checkpoint step, but it was not numerically identical. Its final reward
was -3.40 and average episode length was 34.6. Direct checkpoint restoration
showed an L2 difference of 3.875 between policy parameter trees (the first
policy's L2 norm was 31.104) and 1.228 between value parameter trees (norm
30.246). This is learned-parameter divergence, not merely variable checkpoint
metadata or wall-clock reporting.

MuJoCo Warp has an open upstream request for a mode that guarantees exact
determinism. MotionForge therefore records Warp training as seeded but not
bitwise reproducible. Before selecting the full-training backend, the same PPO
smoke configuration will be run twice with the MJX-JAX implementation. Backend
selection will consider both throughput and numerical repeatability rather
than assuming Warp is appropriate solely because it is faster.

The repeated MJX-JAX smoke run was also not numerically identical. Its policy
parameter tree differed by L2 4.070 from the first JAX run, compared with L2
3.875 between the Warp policies. JAX therefore provided no repeatability
advantage in this test while taking about 135.6 seconds per run versus 42.6
seconds for Warp. Training throughput was approximately 2965 steps/s for JAX
and 5894 steps/s for Warp; evaluation throughput differed by roughly fourfold.

MotionForge will use MJX-Warp for G1 training. Reproducibility for GPU PPO is
defined as pinned code, dependencies, configuration, seeds, and artifacts,
followed by aggregate evaluation across multiple training seeds. It is not
defined as byte-identical learned parameters from repeated runs with one seed.
Deterministic native-MuJoCo evaluation remains the acceptance mechanism for
each fixed exported policy artifact.

Before launching the full 200-million-step training budget, an environment
batch-scaling smoke test will measure memory use and throughput at a more
representative number of parallel worlds.

Brax PPO requires `batch_size * num_minibatches` to be divisible by
`num_envs`. The 128-environment smoke used `32 * 4 = 128`; the 1024-environment
scaling configuration uses `256 * 4 = 1024`. MotionForge validates this
relationship before invoking PPO so invalid launch configurations produce a
descriptive error.

The first validly batched 1024-world attempt completed 102,400 steps and wrote
a checkpoint, but emitted repeated `nefc overflow` warnings. Its inherited
`njmax=90` capacity was below the observed requirement of 93 constraints per
world, meaning constraints could be dropped. The run is rejected as a scaling
result despite its nominal pass summary. Subsequent Warp runs expose contact
and constraint capacities in their Tyro configuration and use `njmax=128` to
provide margin above the observed requirement.

Repeating the 1024-world run with `njmax=128` completed without contact or
constraint-capacity warnings. It processed 102,400 steps, wrote checkpoint
`000000102400`, and sustained approximately 6285 training steps/s. This was
only about seven percent faster than the 128-world smoke, whose PPO minibatch
shape was different. The next capacity test uses Playground's tuned G1 shape
of 8192 environments, batch size 256, 32 minibatches, unroll length 20, and
four updates per batch so its throughput is representative of the intended
full run.

The full-shaped 8192-world Warp smoke completed without contact or constraint
overflows. Peak GPU memory was approximately 28 GiB on the 32 GiB RTX 5090. It
processed 163,840 steps at about 9622 training steps/s and created checkpoint
`000000163840`. This is roughly 53 percent faster than the accepted 1024-world
run and establishes that Playground's tuned batch shape fits on Mayo.

The 100,000-step target still represents only one PPO batch at this shape, and
its resulting KL and reward are not useful policy-quality indicators. Before
committing to 200 million steps, MotionForge will run a 10-million-step
learning sanity check using the same full-shaped configuration. This stage is
intended to catch non-learning, numerical failures, and logging problems; it
is not a replacement for the full training budget.

## Ten-million-step learning sanity run

The seed-0 learning sanity run completed 10,485,760 steps in 102 seconds with
four checkpoints and no non-finite metrics or capacity warnings. After
compilation, measured training throughput reached approximately 368,000
steps/s.

Evaluation reward improved monotonically from -7.20 at initialization to
-4.44, -3.21, -2.99, and -2.70. Training KL fell from 1.00 at the first update
to approximately 0.02 thereafter. Average episode length remained low, ending
at 49.8 of 1000 control steps, so this checkpoint demonstrates a functioning
learning pipeline rather than a usable controller. The configured
200-million-step budget remains necessary.

- `logs/p2/training/g1_standing_learning_10m_seed0/manifest.json`
- `logs/p2/training/g1_standing_learning_10m_seed0/metrics.jsonl`
- `logs/p2/training/g1_standing_learning_10m_seed0/summary.json`
- `logs/p2/training/g1_standing_learning_10m_seed0/checkpoints/`

## Full seed-0 training run

The first full training run completed 202,342,400 environment steps against a
configured target of 200 million. It used the validated full-shaped Warp
configuration: 8192 training environments, 128 evaluation environments, a
1000-step episode limit, 20-step unrolls, batch size 256, 32 minibatches, four
updates per batch, `naconmax_per_env=8`, and `njmax=128`. The run completed in
approximately 705 seconds and reported a final training throughput of about
371,651 steps/s. All recorded metrics remained finite, no capacity-overflow
warnings were observed, and 19 checkpoints were created.

Learning was substantial rather than merely an infrastructure pass. Evaluation
reward rose from -7.20 at initialization to positive values after approximately
42.6 million steps. Average episode length increased from roughly 48 to more
than 800 of 1000 control steps at the best recorded checkpoint.

The best recorded evaluation reward was 15.52 at checkpoint
`000191692800`, whose average episode length was 811.9. The last checkpoint,
`000202342400`, finished with reward 13.60 and average episode length 759.5.
Because the final evaluation regressed relative to the preceding checkpoint,
MotionForge will not equate "last" with "selected." Both checkpoints must pass
the same deterministic command and recovery suite before policy selection.

- `logs/p2/training/g1_standing_full_seed0/manifest.json`
- `logs/p2/training/g1_standing_full_seed0/metrics.jsonl`
- `logs/p2/training/g1_standing_full_seed0/summary.json`
- `logs/p2/training/g1_standing_full_seed0/checkpoints/000191692800/`
- `logs/p2/training/g1_standing_full_seed0/checkpoints/000202342400/`

## Checkpoint restoration smoke test

Checkpoint `000191692800` was restored through MotionForge's Brax 0.14.2
checkpoint compatibility helper. Brax's bundled configuration loader attempts
to resolve optional initializer fields even when they were serialized as
`null`; MotionForge removes only those null overrides so the network factory's
saved defaults apply. No checkpoint parameters are modified.

The restored artifact contained the observation normalizer, policy parameters,
and value parameters. It reconstructed a policy with a 103-element public
observation, a 216-element privileged observation, and a 29-element action.
Repeated deterministic inference from the same observation produced identical
actions. A ten-step MJX-Warp rollout on `cuda:0` remained finite and did not
terminate. The saved network configuration SHA-256 is
`65bb111937d827d19869076926c8c088e9ad24fec2ac3d2b61ba1fbd8dae8bf4`.

- `logs/p2/g1_checkpoint_restore_best_a.json`

This establishes artifact integrity and inference compatibility only. The next
stage evaluates commanded motion, stopping, standing, and recovery over full
episodes before selecting between the peak-reward and final checkpoints.

## Peak-checkpoint command evaluation

Checkpoint `000191692800` completed the first fixed-command suite using four
seeded initial states per scenario, ten seconds per rollout, deterministic
policy actions, and random environment pushes disabled. The first attempt used
`naconmax=8` and emitted narrowphase-capacity warnings requiring at least ten
contacts. That attempt is rejected. The accepted rerun used `naconmax=16` and
`njmax=128`, completed all 32 rollouts without termination, and kept all state
and metric values finite.

Across seeds, the policy's mean local velocities were approximately 0.381 m/s
for a +0.5 m/s forward command and -0.446 m/s for a -0.5 m/s backward command.
For a forward-to-zero command switch at five seconds, mean forward velocity
during the second half fell to 0.036 m/s. Under a constant zero command, mean
velocity remained near zero and all seeds stood for the full ten seconds.

The suite also exposed behavior that must be investigated rather than silently
accepted. A +0.3 m/s lateral command produced only +0.173 m/s, while -0.3 m/s
produced -0.109 m/s. A +0.5 rad/s yaw command produced only +0.119 rad/s, and a
-0.5 rad/s command produced approximately -0.004 rad/s. Thus the peak-reward
checkpoint is stable and can stand, move longitudinally, and stop, but it does
not yet demonstrate adequate bidirectional turning or symmetric lateral
tracking. The final checkpoint will receive the identical suite to determine
whether these deficits are checkpoint-specific.

- `logs/p2/g1_checkpoint_evaluation_best_b.json`

The final checkpoint `000202342400` then completed the same 32 rollouts with
the same seeds and capacities. It also achieved 100 percent ten-second
survival. Its longitudinal behavior was effectively unchanged: mean forward
velocity was 0.387 m/s, mean backward velocity was -0.445 m/s, and its
post-switch stopping velocity was 0.043 m/s. Positive lateral tracking improved
to 0.199 m/s, while negative lateral tracking remained weak at -0.107 m/s.
Pure-yaw tracking remained poor: +0.5 and -0.5 rad/s commands produced mean yaw
rates of only +0.109 and -0.046 rad/s respectively.

The similarity across checkpoints makes ordinary late-training regression an
unlikely explanation for weak turning. The moving-command sampler draws all
three command components independently from continuous uniform distributions.
Consequently, a moving sample with zero forward and lateral velocity but
nonzero yaw has probability zero; the only exactly axis-aligned command added
by MotionForge is the all-zero standing sample. Pure yaw and pure lateral
evaluation therefore probe command combinations that training almost never
presented exactly. Curved-motion diagnostics will test this command-coverage
hypothesis before any reward or training changes are made.

- `logs/p2/g1_checkpoint_evaluation_final_a.json`

## Qualitative checkpoint rendering

MotionForge renders checkpoint rollouts separately from quantitative
evaluation. Policy inference and dynamics remain in MJX-Warp, while selected
states are copied to native MuJoCo for offscreen frames and MP4 encoding. This
keeps rendering overhead and device-to-host transfers out of acceptance
metrics. On the remote Mayo shell, `MUJOCO_GL=egl` is required to create a
headless OpenGL context; this path does not depend on X11 forwarding.

The first video used peak checkpoint `000191692800`, seed 0, and a curved-motion
command of `[0.3, 0.0, 0.5]` for ten seconds. It produced 301 frames at 30 fps,
did not terminate, and retained a minimum root height of 0.745 m. These facts
validate the rendering pipeline and rollout stability. Whether the visible
motion adequately tracks the curve remains a qualitative observation until the
same command is measured by the evaluator.

- `logs/p2/videos/g1_checkpoint_best_curved_left.mp4`
- `logs/p2/videos/g1_checkpoint_best_curved_left.json`

Visual inspection of this rollout found that the turn was jerky and the robot
wobbled back and forth instead of producing clear alternating steps. The
rollout's survival and root-height metrics therefore do not establish useful
curved locomotion. Frame-rate aliasing at 30 fps may affect apparent smoothness,
but it cannot explain the lack of a recognizable stepping pattern. The peak
checkpoint is not accepted as the P2 locomotion prior on this evidence.

The next diagnostic will add curved left/right commands to the quantitative
suite and measure foot-height excursions, contact duty factors, and contact
transitions. This will distinguish weak command tracking from a gait failure or
reward exploit before changing command sampling or reward weights.

The gait diagnostic completed ten scenarios over four seeds without
termination or non-finite values. For ordinary forward, backward, and lateral
motion, each foot underwent approximately 27--31 contact transitions over ten
seconds, contact duty factors were generally near 0.5--0.6, and foot-height
excursions were approximately 0.09--0.14 m. The learned policy therefore has a
repeatable alternating gait rather than relying solely on sliding contacts.

Curved commands also elicited alternating steps. Command `[0.3, 0.0, 0.5]`
produced mean velocities near `[0.208, 0.010, 0.191]`, while
`[0.3, 0.0, -0.5]` produced approximately `[0.216, 0.018, -0.220]`, where the
third component is yaw rate in rad/s. Both directions showed about 25--29
contact transitions per foot, 0.05--0.12 double-support fractions, and roughly
0.10--0.14 m foot-height excursions. Although visual motion remained jerky and
wobbling, the contact data confirms that the curved rollout does contain foot
lifting and alternating contacts.

Pure-yaw behavior is qualitatively different. The +0.5 rad/s condition spent
roughly 56--91 percent of time in double support depending on seed, and the
-0.5 rad/s condition spent 93--100 percent in double support. Several
right-turn rollouts exhibited only three to nine contact transitions per foot
and negligible yaw. This is consistent with the continuous independent command
sampler failing to present pure-yaw examples, while curved commands occupy a
well-sampled part of its distribution. The next environment revision will add
explicit structured command modes and validate their frequencies before any
additional training.

- `logs/p2/g1_checkpoint_gait_diagnostic_peak_a.json`

## Structured command distribution

MotionForge revised only the G1 command sampler; physics, observations, reward
terms, action interpretation, and PPO configuration remain unchanged. The new
distribution allocates 30 percent of samples to standing, 15 percent to pure
forward/backward motion, 10 percent to pure lateral motion, 20 percent to pure
yaw, and 25 percent to mixed three-axis motion. Continuous signed magnitudes
within each active axis retain the original Playground command limits.

A seed-0 test drew 16,384 commands on `cuda:0`. It classified every command
into exactly one mode and observed counts of 4911 standing, 2437 pure-x, 1651
pure-y, 3250 pure-yaw, and 4135 mixed commands. The corresponding probabilities
were 0.2997, 0.1487, 0.1008, 0.1984, and 0.2524, all within the declared
five-standard-error tolerances. All values were finite and within the original
velocity ranges.

- `logs/p2/g1_structured_command_sampling_a.json`

This validates sampling coverage only. A continuation smoke run will restore
the peak checkpoint's observation normalizer, policy, and value parameters and
train under the new distribution. Brax does not store optimizer state in these
checkpoints, so continuation begins with a newly initialized optimizer and
step numbering local to the continuation run; parent-checkpoint provenance must
therefore remain explicit in its manifest.

## Structured-command continuation

A 10,485,760-step continuation restored peak checkpoint `000191692800` and
used the structured command distribution with a reduced learning rate of
0.0001. Evaluation reward under the new distribution improved monotonically
from 15.77 at restoration to 17.82, 18.79, 19.06, and 20.95. Average episode
length rose from 784.9 to 821.1 of 1000 steps, angular-velocity tracking reward
rose from 234.1 to 324.2, and final training KL was 0.014. The run remained
finite and created four checkpoints without capacity warnings.

The final continuation checkpoint passed the same ten-scenario, four-seed
command and gait suite. All 40 rollouts survived ten seconds. Compared with the
parent checkpoint, pure-left yaw increased from +0.110 to +0.292 rad/s and its
double-support fraction fell from 0.749 to 0.076. Pure-right yaw increased in
magnitude from -0.004 to -0.210 rad/s and double support fell from 0.974 to
0.435, although two right-turn seeds still showed partial or near-complete
double-support collapse. Curved yaw improved from +0.191/-0.220 to
+0.240/-0.308 rad/s.

The change did not trade away the established behaviors. Forward tracking
improved from 0.382 to 0.399 m/s, backward tracking from -0.460 to -0.486 m/s,
negative lateral tracking from -0.100 to -0.170 m/s, and post-stop drift stayed
near 0.039 m/s. Standing remained stable and essentially stationary. These
results support additional structured-command continuation, with pure-right
turn consistency as the primary unresolved P2 behavior.

- `logs/p2/training/g1_structured_finetune_10m_seed0_retry/`
- `logs/p2/g1_structured_finetune_10m_evaluation_a.json`

Visual comparison of pure-right-yaw rollouts confirmed the evaluator's
seed-dependent result. Seed 0 executed a little over 180 degrees of rotation in
ten seconds, consistent with its measured mean yaw rate of -0.337 rad/s.
Seed 3 remained stuck standing, consistent with its 0.978 double-support
fraction and -0.029 rad/s mean yaw rate. Thus the remaining failure is an
initial-state-dependent standing attractor, not a general inability to turn or
a rendering artifact.

The next continuation keeps command probabilities, rewards, physics, network,
and learning rate fixed and changes only training duration. This isolates
whether further exposure to pure-yaw samples removes the bad attractor. It
avoids prematurely altering the reward design after the first structured
continuation produced clear improvements.

## Live experiment monitoring

Weights & Biases is integrated as an optional metrics sink in
`motionforge.logging`. MotionForge's local manifest, append-only JSONL metrics,
summary, and checkpoints remain authoritative. The adapter uploads experiment
configuration and scalar progress metrics only; it does not automatically
upload checkpoints or videos. `environment_steps` is defined as the explicit
x-axis so W&B's internal log-call counter is not confused with PPO environment
steps.

The second integration smoke completed 101,120 steps, created a checkpoint,
retained finite local metrics, and displayed approximately 70 Brax training and
evaluation plots alongside GPU telemetry in W&B. W&B is disabled by default
and must be enabled explicitly for monitored runs.

- `logs/p2/training/g1_wandb_integration_smoke_b/manifest.json`
- `logs/p2/training/g1_wandb_integration_smoke_b/metrics.jsonl`
- `logs/p2/training/g1_wandb_integration_smoke_b/summary.json`

## Forty-million-step structured continuation

The monitored continuation restored structured checkpoint `000010485760` and
completed 40,632,320 additional steps in approximately 204 seconds. It retained
the same environment, command probabilities, reward, network, learning rate,
and PPO batch shape as the 10-million-step continuation. Eight checkpoints were
created, all metrics remained finite, no capacity warnings were reported, and
steady training throughput was approximately 373,861 steps/s.

Evaluation was noisy but improved overall. Reward moved from 21.62 at restore
time through intermediate values between 20.25 and 22.88, then ended at the
run's highest recorded value of 24.69. Average episode length increased from
830.6 to 899.0 of 1000 steps. Accumulated angular-velocity tracking increased
from 323.2 to 386.4 and linear-velocity tracking from 691.7 to 762.6. Final KL
was 0.018, with no evidence of numerical divergence.

Checkpoint `000040632320` is the training-metric candidate. It is not selected
until the fixed command/gait evaluation establishes whether pure-right turning
became consistent across initial states and whether standing, longitudinal
motion, lateral motion, and stopping were retained.

- `logs/p2/training/g1_structured_finetune_40m_seed0/manifest.json`
- `logs/p2/training/g1_structured_finetune_40m_seed0/metrics.jsonl`
- `logs/p2/training/g1_structured_finetune_40m_seed0/summary.json`
- `logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320/`

## Evaluation-video callbacks

Training callbacks now live under `motionforge.callbacks`, separate from the
reusable rollout renderer in `motionforge.evaluation` and the W&B transport in
`motionforge.logging`. The G1 video callback receives current policy parameters
from Brax at evaluation boundaries, renders only at a configured environment-
step interval, retains MP4 and JSON sidecar files locally, and optionally logs
the MP4 as W&B media. Videos are qualitative monitoring outputs and do not feed
checkpoint selection metrics.

The callback smoke rendered seed-3 pure-right-yaw rollouts at step zero and
step 101,120, uploaded both to the W&B run, and exposed them through the same
video panel's step history. Training still completed with finite metrics and a
local checkpoint. Video-enabled training uses `MUJOCO_GL=egl` for headless
offscreen rendering.

- `logs/p2/training/g1_video_callback_smoke/manifest.json`
- `logs/p2/training/g1_video_callback_smoke/videos/`

## Forty-million-step candidate evaluation

Checkpoint `000040632320` passed the fixed ten-scenario gait evaluation across
four initial-state seeds. All 40 ten-second rollouts survived and remained
finite. Pure turning became consistent: a +0.5 rad/s command produced a mean
yaw rate of approximately +0.415 rad/s, and -0.5 produced -0.412 rad/s. Every
pure-turn rollout showed an alternating gait with approximately 26--29 contact
transitions per foot and only 0.036--0.052 double-support fraction. The earlier
seed-3 standing attractor was absent.

Other established behavior was retained. Mean forward speed for a +0.5 m/s
command was approximately +0.426 m/s, and mean backward speed for -0.5 m/s was
-0.495 m/s. Lateral commands of +/-0.3 m/s produced approximately
+0.167/-0.184 m/s. Curved commands produced yaw rates of +0.382 and -0.392
rad/s. After the forward-to-zero switch, mean forward speed during the final
five seconds was approximately 0.044 m/s. Standing remained stable and nearly
stationary.

This checkpoint is accepted as the current fixed-command candidate. Additional
training is not justified before recovery evaluation because the specific
command-coverage failure motivating continuation has been resolved without
regressing the other P2 behaviors.

- `logs/p2/g1_structured_finetune_40m_evaluation_a.json`

## Recovery acceptance protocol

Recovery is evaluated separately from fixed-command tracking so that neither
training nor the random-push curriculum can obscure the applied disturbance.
The deterministic evaluator restores the 40-million-step candidate, disables
the environment's random pushes, commands standing, and at 3.0 seconds adds a
1.0 m/s world-frame root-velocity disturbance in each of the forward,
backward, left, and right directions. Each direction is repeated across four
reset seeds for a ten-second rollout.

The report records survival, minimum root height, maximum torso tilt,
post-push speed RMSE, and peak post-push speed. A rollout is classified as
recovered when planar speed remains at or below 0.15 m/s for a continuous
0.5-second window after the disturbance. These initial thresholds are fixed
before examining the result. The top-level pass flag validates execution,
finite state, and complete rollouts; recovery rates remain explicit outcome
measurements rather than being hidden inside an arbitrary infrastructure pass
criterion.

- `scripts/evaluation/evaluate_g1_recovery.py`

The corrected 1.0 m/s evaluation recovered and survived in 13 of 16 rollouts.
All four backward disturbances recovered. Forward, leftward, and rightward
disturbances each recovered in three of four seeds. Successful sustained
recovery took approximately 0.72--2.24 seconds in this run. The three failed
rollouts terminated 0.18--0.92 seconds after the disturbance.

Termination instrumentation established that all three failures were caused
only by Playground's prohibited leg-contact sensors (opposite foot/foot or
foot/shin contact). None involved torso inversion or non-finite state. Two
contact failures nevertheless reached low root heights and torso tilts above
22 degrees, while the rightward seed-1 rollout terminated at only 6.7 degrees
of tilt. Therefore the current 13/16 task-survival result must not be reported
as three physical falls. The next diagnostic will preserve the contact events
but continue control after them, separating transient self-contact from actual
loss of balance before deciding whether recovery-focused training is needed.

The recovery evaluator now provides that contact-tolerant diagnostic by
overriding only the evaluation-time termination predicate. Prohibited contact
timestamps and task survival remain logged, but policy control and physics
continue unless the torso inverts or state becomes non-finite. The existing
checkpoint renderer exposes the same optional planar disturbance and
contact-tolerant behavior, allowing failed task cases to be inspected without
introducing a second rendering implementation.

The contact-tolerant run separated the three original task failures. The
rightward seed-1 disturbance made prohibited leg contact 0.18 seconds after
the push but stayed upright, recovered below the sustained speed threshold in
0.82 seconds, and completed the rollout. Video inspection likewise showed it
as the only previously terminated case that did not fall through the ground.
Forward seed 0 and leftward seed 1 did not recover: they inverted 1.30 and
1.38 seconds after the push, respectively, after reaching approximately 98.4
and 96.4 degrees of torso tilt. Thus physical survival and recovery were 14 of
16 (87.5 percent), while strict Playground task survival remained 13 of 16
(81.25 percent). Backward recovery was four of four and rightward physical
recovery was four of four; forward and leftward recovery were each three of
four.

This diagnostic confirms two genuine recovery failures rather than treating
all prohibited contacts as falls or dismissing them as a termination-rule
artifact. The sample is still too small to tune against individual reset
seeds. A broader fixed disturbance sweep should establish recovery rates by
direction and magnitude before any recovery-focused fine-tuning changes the
training distribution.

The recovery-envelope sweep is fixed in advance at 0.5, 1.0, and 1.5 m/s,
with four cardinal world-frame directions and reset seeds 0--15 at each
magnitude. All other conditions remain unchanged: standing command, push at
3.0 seconds, ten-second horizon, contact-tolerant continuation, 0.15 m/s
sustained recovery threshold, and 0.5-second hold window. This yields 64
rollouts per magnitude and 192 total. Results will be compared using both
physical recovery/survival and strict Playground task survival; the latter
continues to count prohibited leg contact as failure.

The fixed sweep completed all 192 planned rollouts. At 0.5 m/s, all 64
rollouts physically recovered, survived, and avoided prohibited leg contact.
Mean sustained-recovery times were 0.95--1.02 seconds across directions.

At 1.0 m/s, 61 of 64 rollouts physically recovered and survived (95.31
percent), while 59 of 64 met Playground's stricter no-prohibited-contact rule
(92.19 percent). Backward recovery remained 16 of 16. Forward, leftward, and
rightward recovery were each 15 of 16, with mean successful recovery times of
0.95, 1.26, and 1.36 seconds respectively.

At 1.5 m/s, physical recovery and survival were 53 of 64 (82.81 percent), and
strict task survival was 48 of 64 (75 percent). Directional physical recovery
was 14/16 forward, 15/16 backward, 11/16 leftward, and 13/16 rightward.
Successful recovery took 1.42--1.64 seconds on average. The monotonic decline
with disturbance magnitude and the weaker lateral response form a coherent
recovery envelope rather than an isolated seed artifact.

For the P2 locomotion-prior gate, recovery is accepted at the nominal test
level: perfect recovery at 0.5 m/s and at least 95 percent physical recovery
at 1.0 m/s over the declared direction/seed suite. The 1.5 m/s result is kept
as a measured robustness limit for later aggressive-command and push-recovery
baselines; it does not justify changing the locomotion reward before the MVP
controller interface is established. Forward, backward, turning, stopping,
standing, and recovery reproduction are therefore complete. This does not
claim that the controller is maximally robust.

- `logs/p2/g1_recovery_contact_tolerant_a.json`
- `logs/p2/videos/g1_recovery_forward_seed0.mp4`
- `logs/p2/videos/g1_recovery_left_seed1.mp4`
- `logs/p2/videos/g1_recovery_right_seed1.mp4`
- `logs/p2/g1_recovery_sweep_0p5ms.json`
- `logs/p2/g1_recovery_sweep_1p0ms.json`
- `logs/p2/g1_recovery_sweep_1p5ms.json`

- `logs/p2/g1_recovery_1ms_c.json`

## Structured-command continuation smoke

The first continuation restored checkpoint `000191692800` and trained for
10,485,760 additional steps with the structured command distribution. It used
the validated 8192-world Warp shape and a reduced learning rate of `1e-4` to
limit disruption from the freshly initialized optimizer. The run completed in
approximately 103 seconds, created four checkpoints, emitted no reported
capacity warnings, and kept all metrics finite.

Evaluation reward under the new distribution improved monotonically from 15.77
at restoration to 17.82, 18.79, 19.06, and 20.95. Average episode length rose
from 784.9 to 821.1 of 1000 steps. The accumulated angular-velocity tracking
component increased from 234.1 to 324.2, while training KL remained controlled,
ending at 0.014. Because the evaluation distribution changed, these rewards are
compared within this continuation run rather than directly against the original
training run's final reward.

The run validates checkpoint continuation and shows learning pressure in the
intended direction. It is not accepted on aggregate reward alone. Checkpoint
`000010485760` must pass the same fixed ten-scenario command and gait diagnostic
used for its parent before deciding whether more fine-tuning is warranted.

- `logs/p2/training/g1_structured_finetune_10m_seed0_retry/manifest.json`
- `logs/p2/training/g1_structured_finetune_10m_seed0_retry/metrics.jsonl`
- `logs/p2/training/g1_structured_finetune_10m_seed0_retry/summary.json`
- `logs/p2/training/g1_structured_finetune_10m_seed0_retry/checkpoints/000010485760/`
