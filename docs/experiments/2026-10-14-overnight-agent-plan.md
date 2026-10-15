# Overnight diagnosis: active objective and handoff

The user requested continued autonomous monitoring, result compilation and
evidence-driven experiments on g3/g4 until results are reasonable. A persistent
Codex Goal is active in this thread. This is not a guarantee of uninterrupted
service if the agent host/session infrastructure stops or a usage limit is hit.
Slurm jobs continue independently of the chat.

## Active work

### Resumed Oct 15, 13:21 UTC

The agent service stopped at its usage limit; the scheduled Slurm work continued.
On resume, scheduler showed no live jobs and shared artifacts verified all 32
training runs, 640 standard evaluations, and 64 heading validations completed.
No reliable treatment winner was established. See
`2026-10-15-overnight-treatment-results.md` for seed-separated counts and failures.
All 128 saved post-initial checkpoints passed motor-parameter/optimizer freeze
content checks. Follow-up jobs now supersede the historical active list below:

- g3 **4375**, tasks 0–7: final potential .50, GAE 1, init .01, global-normalization
  candidates, seeds 1/2, against the same two outside-panel references at 2 m,
  face-to-face and independent +/-90-degree headings, both action modes.
  Each also saves two role-separated trajectories at a noisy-heading start.
- g3 **4384**, tasks 8–11: matching 4324/4349 controls under the identical protocol.
  These may wait for g3 resources; no unrelated workloads cancelled.
- g4 **4376**: mixed-sampling GPU smoke COMPLETED, exit 0. One training iteration
  plus checkpoint reload and both-role mixed-mode evaluation passed. Metadata
  explicitly says strategy stochastic / motor deterministic.
- g4 **4385**, eight tasks: control vs deterministic frozen motor, paired training
  seeds 0/1/2/3, 100 iterations / 204.8M world steps. Strategy remains stochastic;
  architecture, reward alpha .25, smoothness, opponent mixture, frequencies,
  optimizer, reset geometry and frozen motor weights stay fixed. Standard
  initial/final evaluation uses both homogeneous modes as before; separate
  `mixed_evaluation` files isolate stochastic strategy / deterministic motor.
  No performance conclusion follows from the successful short smoke.

Submission scripts: `scripts/slurm/tag_candidate_validation_g3.sbatch` and
`scripts/slurm/tag_motor_noise_g4.sbatch`. Full g4 job was submitted only after
smoke completion, with `afterok:4376`; no dependency on an expired old array.
Remaining work: inspect new real trajectories, compare modes and both roles,
evaluate transfer, and establish reproducibility before promoting a recipe.

### Latest continuation check, Oct 15 13:43 UTC

The user reaffirmed monitoring and evidence-driven submissions. All eight 4385
tasks reached iteration 23 with finite metrics and zero motor updates. Seven of
eight candidate panels have all eight files; matching controls still await g3
resources, so no new recipe is promoted from partial transfer results.
Calibration 4393 is queued; CPU backfill 4394 has a valid `afterok:4393` dependency.
These Slurm tasks do not require an active chat session to execute.

Important runtime caveat: the goal service still returns `usageLimited` on
`get_goal`, despite this resumed interactive work. Do not claim automatic goal
continuations are active or guaranteed until that service status is cleared.
The sub-agent is performing a bounded live monitor now; its existence is not
proof of indefinite autonomous service. Preserve this handoff and inspect actual
scheduler/artifact state on the next continuation; never duplicate jobs solely
because the agent was interrupted.

### Historical first-stage work (now completed)

- g4 array 4324: control, no smoothness, alpha .50, GAE 1.0; seeds 1/2.
- g3 array 4340: small command initialization .01/.10, lower LR, small init
  without smoothness; seeds 1/2. Smoke 4339 passed.
- Motion audit 4323 completed 88 probes and uploaded W&B artifacts.
- g4 queued smoke 4348 and dependent accumulation array 4349: control,
  valid-timestep weighting, global advantage normalization, both; seeds 1/2.
- g3 queued smoke 4354 and dependent heading-diversity array 4355: control,
  independent initial heading noise +/-30 or +/-90 degrees, and +/-90 degrees
  with small initialization; seeds 1/2. Nonzero-heading GPU validation pending.
- A monitoring sub-agent checks scheduler/log progress. Tested CPU-only
  snapshot and matched-evaluation compilers are available; parent owns decisions.
- Latest checked training records were finite across all 16 active runs.
- At 03:44 UTC Oct 15, 4324 reached iterations 53–54 and 4340 reached 49–50;
  all 16 were RUNNING. Follow-up smokes wait for resources, arrays for dependencies.
  No final evaluation data yet; no treatment winner established.
- Saved checkpoint content audit passed 24 checkpoints: frozen motor parameters,
  optimizer state, and zero update counters agree with each run's iteration 0.

## Working loop

1. Check scheduler, error markers, metrics finiteness, updates and log age.
   Training completion is distinct from evaluation completion. Do not resubmit
   a job solely because it has moved from training into evaluation.
2. Compile fixed-reference initial/final evaluations separately by role,
   opponent, distance and deterministic/stochastic execution. Never use the
   success-only Elo curves or variable-opponent training reward as readiness.
3. Compare candidate mechanisms across both discovery seeds; audit actual
   commands and radial motion, not just appealing render clips.
4. Promote a promising arm to additional training seeds and reserved opponents
   before describing it as a robust result. A candidate that only improves
   one role, has both agents cooperate in a tag, or trades tags for OOB is not
   a solved competitive task.
5. If no arm improves, use initial/final command distributions, fixed-command
   tracking and strategy gradients/credit diagnostics to choose the next
   isolated test. Maintain a tested control and preserve failed artifacts.
6. Persist conclusions, uncertainties, commands and job IDs so the work can
   resume from this checkout if the session stops.

## Evidence standard and constraints

"Reasonable" means reproducible improvement rather than a single good video:
working pursuit beyond discovery opponents/start distance, no systematic
circle/retreat failure, low physical failure rates, and competent evasion
measured separately against fixed pursuers. Additional seeds and independent
opponents are required; two discovery seeds alone do not establish robustness.

At most the 16 user-provided GPUs, scheduled through Slurm. No unrelated jobs
cancelled, no checkpoint/log deletion, no Elo-based curriculum substitutions,
no fabricated skill/difficulty curves. Preserve TAG payoff, truncation semantics
and known-good locomotion while diagnosing strategy. Any substantive deviation
must be a named ablation, not an unlogged default change. Do not blindly expand
compute or endlessly repeat a failed recipe. Escalate missing access, scope
changes or required user choices explicitly.

Current experiment details and measured circling findings:
`docs/experiments/2026-10-14-circling-and-backward-motion-audit.md`.
Compiled live reports: `logs/hierarchical/diagnostics/overnight/`.
