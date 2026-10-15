# Reset diversity: world rotation is not local steering diversity

## Confirmed CPU audit

`scripts/evaluation/audit_tag_reset_diversity.py` uses actual reset sampling,
native MuJoCo forward kinematics, and production observation adapters across
128 seeds. At fixed 1.5 m separation, globally rotated face-to-face starts have:

- world-position coordinate standard deviation about .53 m;
- maximum strategy-feature standard deviation **2.41e-8**;
- target-bearing standard deviation approximately 4.6e-8 / 1.3e-7 radians for
  the two robots.

So reset seeds change world orientation, but essentially not the initial local
strategy input. Independent heading perturbations of +/-pi/2 on the same native
spawns produce local target-bearing standard deviations .889/.912 radians.

Output: `logs/hierarchical/diagnostics/overnight/reset_diversity.json`.
This is an initial-state observation audit, not evidence that entire rollouts
are identical: action noise, contact and square boundaries can diversify later
states. It identifies a plausible training/generalization limitation, not a
proven sole cause of pursuit failure.

## Opt-in feature and compatibility

- `TagResetConfig.heading_noise_radians` and the corresponding Warp config
  default to **0**. The original angle/separation RNG streams are unchanged.
- Nonzero noise independently perturbs each robot's body heading with a
  separate folded-in RNG stream, preserving positions, velocities, joint pose,
  terrain and separation. Quaternions remain normalized. Range is [0,pi].
- Training/evaluation CLI key: `initial_heading_noise_radians`.
- Standalone plot/render tools accept an explicit override or honor the saved
  training setting when `use_training_reset=true`; zero is used for old runs.
- No active experiment was restarted or had its reset distribution changed.
- Three new CPU tests passed, including an exact pre-change reset SHA256
  fixture, deterministic noisy sampling, unchanged non-heading state, bounds and
  Warp config validation. Existing native reset regression also passed.
- Nonzero-noise GPU training/evaluation smoke is still pending; do not claim
  GPU validation until job 4354 succeeds.

## Queued follow-up

- Smoke **4354** on g3, pending resources. Exercises noisy vmapped training and
  noisy checkpoint evaluation, not merely default zero-noise behavior.
- Array **4355** on g3, eight tasks, gated on `afterok:4354`.
- Four arms, training seeds 1/2 each: unchanged face-to-face control, +/-30-degree
  heading range, +/-90-degree heading range, +/-90-degree plus command-head
  initialization scale .01 (explicit interaction, not a single-factor claim).
- Same fixed 1.5 m training distance, frozen accepted locomotion, TAG payoff,
  alpha .25 potential, smoothness .005, strategy LR, GAE .95 and league mixture.
- Standard initial/final development panel remains face-to-face for all arms.
- Additional final validation uses two independent reserved opponents at 2 m,
  seeds 5000–5015, zero versus +/-90-degree heading perturbations, both action
  modes and both learner roles. These files live under each run's `validation/`
  and must be compiled separately from the 20 standard `evaluation/` files.
- W&B group `tag_heading_diversity`; videos disabled for the training sweep,
  saved checkpoints support later matched trajectory/video audits.

Do not promote a policy based only on its face-to-face success. The purpose is
to learn steering rather than a favorable open-loop start command, and to test
that on independent opponents and broader starts. Record failures honestly.
