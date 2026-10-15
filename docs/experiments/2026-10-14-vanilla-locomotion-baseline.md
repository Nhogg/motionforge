# Noncompetitive locomotion continuation baseline

## Question

Does additional standalone command-following PPO improve the accepted locomotion
prior, or do held-out tracking and recovery metrics plateau without an adaptive
competitive task? A plateau is a hypothesis, not a required result.

## Protocol

- Three training seeds (0, 1, 2), each on one GPU on g4.
- Initialize all seeds from the same accepted checkpoint:
  `logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320`.
- Train for 200M additional environment steps per seed, with approximately 40M
  checkpoint spacing (Brax rounds to complete training batches).
- Existing standalone `train_g1_standing.py`; unchanged locomotion objective,
  observation/model architecture, and stationary structured command sampler.
- Match the prior's training settings: 8192 environments, 128 evaluation
  environments, 20-step unroll, batch size 256, 32 minibatches, four updates,
  learning rate 1e-4, episode length 1000. Existing random pushes remain enabled.
- Command probabilities: standing .30, axial-x .15, axial-y .10, yaw .20,
  mixed .25. Optional rapid-command training is disabled. No strategy, opponent,
  TAG reward, league, Elo, or adaptive difficulty is introduced.
- Brax restores network/normalizer parameters, not a complete optimizer-state
  continuation. There is one fresh optimizer at the beginning, not periodic
  optimizer restarts.

This is a **pretrained continuation baseline**, not locomotion learned from zero.
It does not independently demonstrate end-to-end TAG learning from scratch.
Environment steps here simulate one robot; TAG world steps simulate two robots.
Do not describe equal world-step budgets as equal motor compute. A future joint
comparison also needs matched update budgets, rewards, and evaluation conditions.

## Evaluation

After training, evaluate the initial prior and every saved checkpoint with the
same reset seeds 3000–3003, deterministic actions, 10-second trajectories, and
random pushes disabled. Reuse the existing fixed command and abrupt reversal
evaluators. Report linear/yaw tracking RMSE, survival, and transition recovery.
Inspect errors together with survival because an early failure can shorten the
interval over which tracking error is measured. The two existing suites use
different termination criteria; compare within each suite, not their absolute
survival rates against one another.

Raw JSON is retained under each run's `fixed_benchmark/`; `curve.json` aggregates
the learning curve. Training W&B group: `g1_vanilla_fixed_commands`. Fixed
benchmark W&B group: `g1_vanilla_fixed_benchmark`, project `luc-ssl/MotionForge`.
Benchmark curves upload after the evaluation pass completes. Benchmark survival
or recovery failures are observations, not reasons to discard a checkpoint;
missing artifacts or nonfinite measurements fail the job.

No bounded sigmoid or fabricated Elo is used as evidence of a plateau. Fixed
task difficulty is constant by design and is not a learned difficulty result.
To support a plateau claim, inspect late-checkpoint changes across seeds and
task-wise raw errors; extend training if improvement is still clear. Run future
competitive locomotion checkpoints through this exact benchmark too.

## Submitted jobs

- GPU smoke: **4314**, one short training-and-evaluation pass on g4.
- Full baseline array: **4315**, three seeds, `afterok:4314`.
- Both were submitted while TAG array **4306** occupied g4. These are queued,
  not verified GPU-running baselines. A failed smoke leaves the full array
  dependency-blocked intentionally; inspect its logs before resubmitting.
- Existing TAG runs and interactive allocation 4290 on g3 were left untouched.

Launcher: `scripts/slurm/g1_vanilla_fixed_commands_g4.sbatch`.
Benchmark driver: `scripts/evaluation/evaluate_vanilla_locomotion_curve.py`.
Logs: `logs/slurm/g1-vanilla-{smoke,fixed}-*.{out,err}`.
Outputs: `logs/vanilla/training/g1_vanilla_fixed_seed*_job*/`.

## TAG next decision

Let 4306 finish its four paired seeds (potential alpha=.25 versus disabled
shaping, frozen locomotion). Decide using held-out opponent and self-play TAG
rates in both action modes, distance closed/minimum distance, time near contact,
and failure causes. Compare reward component sums to terminal magnitude. Training
batch TAG rates alone depend strongly on the sampled opponent and are not a
generalization result. If approach improves without contact, inspect trajectories
and final-contact behavior before adding another reward. If approach does not
improve, inspect command execution, boundary behavior, and reward/regularizer
scales. Reintroduce joint motor learning only after a reliable strategy baseline
is established; then compare both motor policies with the fixed benchmark.
