# Independent motor sampling diagnostic

## Evidence motivating this comparison

The completed 32-run sweep did not identify a robust both-role treatment.
Separately, the paired native CPU experiment showed replacement motor noise
increases standing yaw-rate RMSE (.714 vs .270 original P2 vs .022 deterministic).
These are instantaneous fluctuations, not proof of sustained turning bias.
Original P2 noise is not uniformly better under combined commands. The learned
circle/retreat policies also explicitly request their yaw/backward commands.

We therefore test removing motor exploration while keeping strategy exploration.
This does not restore the original P2 distribution or claim it caused failure.

## Implementation and safeguards

`HierarchicalRolloutConfig.stochastic_locomotion_actions=None` retains the existing
joint sampling switch exactly. An explicit boolean affects only motor sampling;
strategy sampling and PRNG splitting are unchanged. The isolated
`train_tag_motor_noise_ablation.py` requires frozen locomotion for its deterministic
motor arm. No architecture, motor reward, PPO target, role routing or weight change.

Three CPU tests include the actual jitted runner step with a fixture environment:
default equivalence, unchanged strategy actions/PRNG, changed motor sampling,
freeze guard, wrapper restoration, and truthful mixed-mode evaluation metadata.
Renderer role checks and Ruff pass. Callback metadata and frame annotations now
identify the two action modes separately when they differ. Old defaults remain
homogeneous. Standalone plot/render tools still use their original joint switch;
do not label their homogeneous stochastic renders as this mixed-mode condition.

GPU smoke **4376** completed with exit 0, finite training metrics, saved/reloaded
checkpoint, and valid both-role mixed-mode evaluation. Its short .1-second run is
execution validation only, not evidence of tracking or TAG competence.

## Training study

Array **4385**, g4, seeds 0/1/2/3 in each of two arms:

- Control: stochastic strategy and motor actions during training.
- Deterministic frozen motor: stochastic strategy actions, deterministic motor.

Both use fresh strategy and the same accepted P2 motor, frozen throughout;
100 iterations, 4096 environments, horizon 500 / 10 seconds, 204.8M world steps;
potential alpha .25, smoothness .005, historical probability .75, 1.5 m starts,
same optimizer/architecture/frequencies. No combination with other new treatments.

W&B group: `tag_motor_noise_diagnostic`.
Paths: `logs/hierarchical/training/tag_motor_noise_{control,deterministic_motor}_seed{0,1,2,3}_job4385`.

## Evaluation and decision

At initial/final checkpoints, preserve earlier 20-file standard evaluation:
two development references, 1.5/2 m, deterministic and stochastic joint execution,
16 reset seeds, both learner roles, plus self-play as a separate diagnostic.
Add ten separate mixed-mode files under `mixed_evaluation`, same conditions but
strategy stochastic / motor deterministic. Never pool those action modes.

Primary question: does removing frozen-motor noise improve pursuit reproducibly
across training seeds without evader OOB/TAG collapse, and does the improvement
survive changing evaluation action mode? Compare each treatment to the same-seed
control. Four training seeds remain a small exploratory sample, not a guarantee.
Historical references are training/development opponents; independent-reference
and broader-start evaluation is still required before promotion. Self-play tags
and the known-flat aggregate Elo curves are not competence-selection criteria.

The g3 arrays 4375/4384 separately test existing candidates and matching controls
against outside-panel references; they do not evaluate new 4385 policies yet.
