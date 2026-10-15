# Strategy gradient accumulation audit

## Confirmed implementation behavior

The existing hierarchical updater partitions contiguous samples into chunks,
normalizes advantages separately inside each chunk, then averages chunk means
and gradients before one optimizer step. If valid sample counts differ because
episodes end at different times, this is not the global valid-timestep objective.
An empty chunk also contributes a zero to the chunk-average denominator.

In an actual strategy actor/value PPO fixture with unequal padding, the maximum
gradient-leaf difference between the existing chunked and full-batch objectives
is approximately **0.63413**, with advantage normalization disabled to isolate
weighting. This is a controlled synthetic test, not an estimate of the discrepancy
on a training rollout and not proof that this causes the pursuit failures.

## Isolated implementation

`motionforge/training/tag_strategy_objective_ablation.py` contains an optional
updater adapter. Global normalization is performed once over valid samples; each
chunk loss/gradient receives multiplier:

    number_of_chunks * valid_samples_in_chunk / total_valid_samples

The inherited accumulator's division by chunk count then yields the global
valid-sample mean. Strategy-only: locomotion loss and optimizer remain unchanged.
Production updater defaults and active jobs 4324/4340 were not edited.

Nine CPU tests passed with actual strategy losses and all gradient leaves:
unequal padding, normalization on/off, empty chunks, all invalid, perturbed
policies exercising clipping, jitted gradients, unmodified control behavior,
and the locomotion non-interference guard. Both loss and gradient accumulation
are checked against a full-batch reference at chunk sizes 2/4/8.

This chooses a **valid-timestep** objective, not equal weighting of episodes or
roles. Longer trajectories contribute more samples. Any behavioral improvement
must be attributed to the named experimental factors until the factorial results
separate them; neither objective definition alone is a competence claim.

## Queued factorial study

- GPU smoke **4348**, g4, waiting for resources; exercises the `both` adapter in
  a one-iteration training pass. Not yet validated on GPU.
- Array **4349**, g4, eight one-GPU tasks, gated on `afterok:4348`.
- Four arms: unchanged control, valid-count weighting only, global advantage
  normalization only, both. Each uses training seeds 1 and 2.
- Same frozen accepted locomotion, alpha .25 potential, smoothness .005,
  historical mixture, strategy learning rate, GAE .95, architecture and task.
- 100 iterations / 204.8M steps; saved checkpoints at 0/25/50/75/100.
- Same initial/final fixed development-opponent tests at 1.5/2.0 m, deterministic
  and stochastic execution, 16 reset seeds, both learner roles, plus self-play.
- W&B group `tag_accumulation_diagnostic`; no training videos in this follow-up.
- No current jobs cancelled or restarted. This uses g4 as existing tasks finish.

Launcher: `scripts/slurm/tag_accumulation_g4.sbatch`.
CPU regression: `scripts/tests/test_tag_strategy_accumulation.py`.
The overnight snapshot monitor includes array 4349 once its run directories exist.
