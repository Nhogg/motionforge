# Experiment ledger

Use one entry per isolated hypothesis. Record the protocol before launch, Slurm
IDs/output paths, common evaluation conditions, observations and the resulting
decision. Keep training-seed variation separate from evaluation-reset variation.
Do not select checkpoints using shaped training return or a single render.

## E001 — Distance-potential shaping (completed 2026-10-17)

- Question: does potential shaping improve pursuit without encouraging hurling?
- Runs: 4917–4925; `logs/mofo/shaping_w<weight>_s<seed>-<job>/`.
- Matrix: weights 0 (disabled), 0.25, 1.0; seeds 0/1/2; 1000 PPO updates,
  4096 physical worlds, 24 steps/update, 98.304M physical transitions.
- Shared protocol: stage-1 `logs/mofo/stage1_snapshot/model_17800.pt`, contact
  TAG, +/-10 TAG payoff, individual -10 fall/OOB penalties, healthy 20 s timeout
  as evader win, gamma .99, 50% self-play, PFSP snapshots every 100 updates.
- Evaluation: 256 first episodes per checkpoint/matchup, reset seed 40000,
  random learner role (130 pursuer/126 evader), deterministic actors, no
  observation noise, independent 1 s post-TAG survival audit. Self-play uses the
  same checkpoint for both seats; scripted uses the fixed stage-1 chase/flee
  controller. Evaluation jobs 4955–5001, all completed.
- Coverage: all nine policies at iteration 250 and 999 against both opponents;
  all nine at 500 in self-play; pre-update baseline evaluated separately.
- Machine-readable per-run results and three-seed means:
  [E001_shaping_results.json](experiments/E001_shaping_results.json).
  Individual JSONs retain every trial in each run's `eval/` directory.

Controlled TAG = valid pursuer TAG with **both robots surviving the following
one second**, divided by all learner-as-pursuer episodes. It is not a force or
safety certification. Numbers below are means across three training seeds.

| Weight | Self-play controlled TAG, 250 | Fixed-opponent controlled TAG, 250 | Self-play controlled TAG, 999 |
|---|---:|---:|---:|
| 0 | 64.6% | 23.1% | 20.3% |
| 0.25 | 61.8% | 76.4% | 26.9% |
| 1.0 | 69.0% | 51.5% | 26.4% |

Findings:

- Final raw pursuer TAG rates are approximately 100%; every final policy has
  zero healthy-timeout evader wins in self-play. This is not balanced TAG mastery.
- Contact control deteriorates with training. For weight 0/seed 2, iteration
  250 -> 999: mean pursuer contact closing speed 0.34 -> 2.86 m/s, peak contact
  force 40 -> 270 N, controlled self-play TAG 96.2% -> 30.8%.
- The apparent no-shaping seed-2 champion at 250 only achieves 10.8% controlled
  TAG against the fixed opponent: self-play alone overstates generality.
- Best individual candidate across both pursuit evaluations: weight 1/seed 0,
  iteration 250 (`shaping_w1.0_s0-4923/model_250.pt`): 76.2% self-play and 74.6%
  fixed-opponent controlled TAG. Evasion is still weak.
- Weight .25 is the most promising reproducible fixed-opponent setting at 250.
  Three seeds are limited evidence; this is a selection result, not a final
  independent paper test.

Decision: preserve early checkpoints; do not extend the immediate-reset runs
unchanged. Test post-TAG recovery separately. The proposed causal explanation is
that immediate reset hides later falls; E002 tests this explanation rather than
assuming it has been proven. Evader learning remains a separate unresolved issue.

## E002 — Post-TAG recovery training (protocol frozen 2026-10-17)

- Hypothesis: exposing post-contact falls to the existing own-failure penalty
  reduces hurling without changing the valid TAG payoff.
- Matrix: `rules.post_tag_s=0.0,0.5,1.0`, each with seeds 0/1/2.
- Shared config: `+experiment=recovery` inherits E001, shaping weight .25;
  same stage-1 initialization, PPO settings, league, contact rule, regularizers,
  architecture, and 1000-update physical-transition budget. Fresh optimizers.
- TAG pays +/-10 **once, immediately at valid first contact**, including during
  recovery arms. No repeated-contact reward and no extra opponent-failure bonus.
- Recovery arms continue both policies and normal individual regularizers;
  a fall/OOB still terminates and penalizes only the offender. A later failure
  does not erase the recorded TAG from league game outcomes. The new metrics
  distinguish `post_tag_failure` from `post_tag_survived`.
- Distance potential closes once at TAG and is zero throughout recovery.
  Healthy non-TAG timeout remains 20 s. A TAG just before the deadline receives
  the full recovery window. No new phase observation or network input is added;
  the actor must react using its existing observations. This is a limitation of
  this minimal compatibility-preserving experiment.
- Fairness: equal physical-transition budget, not equal completed-game count;
  recovery occupies simulator steps, so game counts can differ.
- Evaluation is standardized independently of the training grace: fixed 1 s
  recovery audit, held-out reset seed 40000, 256 first episodes, random roles,
  deterministic policies, self-play and scripted opponents, iterations 250/999.
  This reuses the E001 selection set; reserve a new reset seed for final claims.
- Selection: controlled TAG against both opponents, contact speed/force/tilt,
  failure rates and evader timeout wins, separately by seed. A setting does not
  pass solely because raw TAG rate improves. Best mean controlled TAG is primary;
  check for loss of pursuit coverage and unresolved evasion.
- Validation: inline synthetic assertions cover one-time payoff, repeated
  contact, timing, simultaneous failure precedence, potential closure, healthy
  timeout and late TAG. GPU warm-start/PPO/render smoke 5002 completed; artifacts
  in `logs/mofo/recovery_smoke-5002/`. No temporary test script retained.
- Source base: `01c7ace9866d452372f2850510ebfb3b7f4057f5` plus the recovery patch;
  each training run records its source diff in `git/motionforge.diff`.
- Launch manifest: [E002_launch.json](experiments/E002_launch.json). Submitted
  through `bash scripts/slurm/prod.sbatch uv run ...`, one GPU per training run,
  on g4. Eight training jobs verified running; the ninth waits for resources.
  Thirty-six held-out evaluation jobs wait on verified `afterok` dependencies,
  four per training run (250/999 x self-play/scripted). W&B group:
  `MotionForge / recovery_antihurl`. Checkpoint videos remain enabled.
- Submission hiccup: jobs 5003–5047 were the failed/cancelled first attempt;
  non-interactive SSH lacked uv on PATH and Slurm ignored the dependency env
  variable. The template now validates uv and translates dependencies explicitly;
  replacement training uses the explicit `.venv-mjlab` environment. These failed
  jobs are not scientific experiment results and did not complete training.
- Results pending. No claim that this has solved hurling or evasion yet.

| Recovery seconds | Seed 0 training job | Seed 1 training job | Seed 2 training job |
|---|---:|---:|---:|
| 0.0 (control) | 5048 | 5053 | 5058 |
| 0.5 | 5063 | 5068 | 5073 |
| 1.0 | 5078 | 5083 | 5088 |
