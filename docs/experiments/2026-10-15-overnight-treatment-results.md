# Overnight TAG treatment results — 2026-10-15

## Conclusion

Some individual policies now pursue successfully, but **no treatment reliably solves both roles across both training seeds and evaluation conditions**. Do not select a winner from self-play tags or training reward. Stronger potential shaping deserves a controlled follow-up; it is not a demonstrated general solution. Heading diversity exposes generalization failures rather than curing them in this sweep.

This is a retrospective, exploratory comparison of arrays 4324 (mechanism), 4340 (initialization), 4349 (accumulation), and 4355 (heading diversity). Every arm has only **two training seeds, 1 and 2**.

## Evidence and definitions

- All 32 runs have their exact 20 standard evaluation files, including initial checkpoint 0 and final checkpoint 100. All passed execution/finite checks and supplied the expected per-role samples.
- All eight heading runs additionally have eight final reserved-reference validation files. The comparison compiler confirms completion independently of training completion.
- Total: 640 standard files and 64 validation files, each with 16 rollouts in each role: 22,528 observed role episodes. These are **not** 22,528 independent training replicates.
- Pursuer benchmark: valid TAG count divided by **all** observed episodes, including truncations and physical failures.
- Evader benchmark: tag-free `TIME_LIMIT_TRUNCATION` with physical survival divided by **all** observed episodes. Opponent failure is not full-window completion. This is a 10-second finite-window benchmark, not a terminal game win or payoff.
- `role_success` can be null on censored trajectories and is not used as the denominator or success label here.
- Standard references are historical checkpoints 005 and 020, which were in the training league. Two references × 16 reset seeds gives 32 episodes per training-seed/distance/action-mode/role cell below. Counts are combined within a training seed for compactness; opponent-separated counts remain in the machine-readable report.
- “Stochastic” here samples **both** strategy and motor actions; deterministic samples neither. These panels cannot isolate motor-noise effects.

Source: `logs/hierarchical/diagnostics/overnight/comparison.json`, refreshed with:

```bash
python scripts/evaluation/compare_tag_mechanisms.py
```

The accompanying `comparison.md` contains every individual reference and seed. Raw evaluation JSON was also checked, including reserved heading090 seed2: correct checkpoint 100, new reset seeds 5000–5015, separation 2 m, supplied heading noise, and all execution checks true.

## Standard historical pursuit: distance and action mode matter

Each table entry is **[TAG count at 1.5 m, TAG count at 2.0 m]**, with **32 episodes at each distance**. Training seeds remain separate. All deterministic initial pursuer counts were zero in these cells; stochastic initial counts ranged from 0 to 10/32 at 1.5 m and 0 to 2/32 at 2 m. Exact initial/final pairs are in `comparison.json`.

| Array / treatment | Seed 1 deterministic | Seed 2 deterministic | Seed 1 stochastic | Seed 2 stochastic |
|---|---|---|---|---|
| 4324 control | [16, 0] | [16, 26] | [23, 3] | [14, 11] |
| 4324 no smoothness | [32, 16] | [32, 32] | [31, 20] | [14, 10] |
| 4324 potential alpha 0.50 | [32, 32] | [16, 16] | [32, 32] | [15, 14] |
| 4324 GAE lambda 1.0 | [32, 32] | [0, 0] | [32, 32] | [0, 0] |
| 4340 command-head init 0.01 | [32, 32] | [28, 0] | [32, 31] | [14, 4] |
| 4340 command-head init 0.10 | [32, 32] | [32, 0] | [32, 32] | [18, 7] |
| 4340 init 0.01, no smoothness | [32, 32] | [0, 0] | [31, 23] | [6, 0] |
| 4340 learning rate 1e-4 | [16, 0] | [16, 0] | [14, 0] | [9, 3] |
| 4349 control | [16, 0] | [16, 32] | [22, 2] | [15, 9] |
| 4349 global normalization | [32, 31] | [0, 0] | [32, 26] | [13, 8] |
| 4349 valid-sample weighting | [6, 0] | [0, 0] | [8, 0] | [7, 11] |
| 4349 both accumulation changes | [0, 0] | [0, 0] | [8, 0] | [9, 15] |
| 4355 control | [16, 0] | [18, 31] | [22, 2] | [15, 9] |
| 4355 heading ±30 degrees | [16, 0] | [27, 16] | [18, 0] | [13, 8] |
| 4355 heading ±90 degrees | [16, 0] | [0, 0] | [13, 0] | [1, 1] |
| 4355 heading ±90, init 0.01 | [32, 32] | [0, 0] | [32, 29] | [4, 0] |

Interpretation:

- Potential alpha 0.50 gives the best balanced **development candidate**: seed1 pursuer 64/64 and evader full-window 64/64 in both modes. Seed2 evader also completes 64/64, but pursuer reaches only 32/64 deterministic and 29/64 stochastic. Its deterministic seed2 pursuit is below the same-array control's 42/64. There is no consistent treatment advantage across seeds/modes.
- GAE 1.0 and global normalization produce strong seed1 policies but fail deterministic seed2 pursuit completely. The valid-weighting and combined accumulation changes do not rescue pursuit. Full-valid-sample objective equivalence is a correctness/property result, not a promise of better learning.
- Small command-head initialization improves seed1 pursuit, but seed2 frequently loses the benefit at 2 m and can damage evasion. Smaller learning rate alone is not a rescue.
- The same-named controls in different arrays are not numerically identical outcomes. Use the matching array's control; do not pool controls as additional independent training seeds.

## Pursuit-only scoring would conceal evader collapse

Important final fixed-historical examples, combined across the two distances within one training seed (64 episodes per role/mode):

| Run | Pursuer TAGs deterministic / stochastic | Evader full-window deterministic / stochastic | Explanation |
|---|---|---|---|
| 4324 no smoothness, seed2 | 64/64; 24/64 | 0/64; 16/64 | Evader has 64/64 deterministic OOB and 48/64 stochastic OOB; deterministic evader physical survival is zero. |
| 4340 init 0.01, seed2 | 28/64; 18/64 | 9/64; 24/64 | Deterministic evader gets tagged 48 times and exits 7 times. |
| 4355 heading ±30, seed1 | 16/64; 18/64 | 10/64; 23/64 | Evader OOB in 54/64 deterministic and 41/64 stochastic games. |
| 4355 heading ±90 + init0.01, seed1 | 64/64; 61/64 | 14/64; 19/64 | High pursuit coexists with evader TAGs and OOB: deterministic 32 TAGs, 18 OOB. |

Do not call an opponent's fall/OOB a TAG or an evader victory. High physical survival alone is also insufficient: an upright evader can still be tagged.

## Heading experiment: reserved-reference validation

These are final-only tests at **2 m**, new reset seeds 5000–5015, against two references outside the configured historical training panel:

- `tag_clean_start_mixed_4096_seed4_g4`, checkpoint 950 (“baseline”).
- `tag_ablation_historical_050_strategy_lr_1e3_seed2_job4004`, checkpoint 500 (“ablation”).

Each cell below is **Pursuer TAG count; Evader tag-free full-window count**, each out of **32**. It combines references only within one training seed, heading distribution, and mode. ±90 means independent uniform heading perturbations around the original face-to-face headings, not arbitrary randomized positions.

| Training arm / seed | Face-to-face deterministic | Face-to-face stochastic | ±90 headings deterministic | ±90 headings stochastic |
|---|---|---|---|---|
| Control / 1 | 32; 17 | 32; 26 | 15; 23 | 15; 22 |
| Control / 2 | 0; 32 | 2; 32 | 5; 22 | 5; 25 |
| ±30 / 1 | 32; 0 | 32; 8 | 13; 4 | 13; 7 |
| ±30 / 2 | 0; 32 | 1; 32 | 8; 22 | 7; 21 |
| ±90 / 1 | 32; 12 | 31; 16 | 13; 15 | 12; 19 |
| ±90 / 2 | 16; 32 | 20; 22 | 6; 23 | 7; 17 |
| ±90 + init0.01 / 1 | 32; 6 | 30; 7 | 15; 6 | 12; 5 |
| ±90 + init0.01 / 2 | 16; 0 | 19; 5 | 10; 12 | 9; 14 |

Findings:

- Heading ±90 improves seed2 face-to-face reserved pursuit versus control, but the deterministic 16/32 improvement is entirely against **baseline** (16/16), with **0/16 against ablation**. Stochastic splits are baseline 14/16 and ablation 6/16. Do not interpret the aggregate as broad opponent mastery.
- The same ±90 seed2 policy has **0/64 deterministic development TAGs**, only 2/64 stochastic development TAGs, and just 6/32 deterministic / 7/32 stochastic reserved TAGs under randomized headings. This is opponent- and reset-dependent transfer, not a robust cure.
- Heading ±30 seed1 evader has **32/32 OOB** in face-to-face deterministic reserved tests. Its zero full-window score is not a bookkeeping artifact.
- Combining ±90 heading noise with small initialization trades roles badly. Seed2 gets 16/32 deterministic reserved pursuit but evader 0/32 full-window completion; all 32 evader episodes end in TAG.
- Increasing reset diversity alone does not demonstrate reliable pursuit on the diversified test distribution. Opponents also experience those perturbed starts, so changes in learner TAG rate do not isolate learner skill from opponent difficulty.

## Protocol and claim limits

1. Standard historical tests are **development** tests against training-league opponents. Reserved references were not in the configured historical panel but have been used in earlier development; “outside this training panel” is supported, “untouched independent test set” is not.
2. Only the heading array received reserved validation. A stronger-potential policy cannot yet be ranked against a heading policy on that benchmark.
3. Initial face-to-face resets vary global rotation but offer little egocentric strategic diversity. Sixteen deterministic reset trials are not sixteen substantially different situations, and never replace independent training seeds.
4. Validation has final checkpoints only. It supports endpoint comparisons against the same-array control, not initial-to-final validation learning curves.
5. Evaluation uses a 10-second collector horizon while the environment episode-duration configuration is 20 seconds. Collector exhaustion is correctly a truncation; the benchmark is **survive this 10-second observation window**, not win a 20-second game.
6. Self-play changes both agents. For example, init0.01 seed2 has 16/16 deterministic self-play TAGs yet 0/32 development pursuit TAGs from 2 m. Self-play TAGs establish neither robust pursuit nor evader success.
7. Root-center distance is not contact geometry. A valid limb/body TAG can occur above 0.5 m root separation; `time_below_0_5m` is a proximity diagnostic, not a necessary TAG condition.
8. No policy-selection threshold or statistical superiority claim was preregistered. Counts here are descriptive, with only two training seeds per treatment and multiple exploratory comparisons.

## Recommended next tests

1. **Separate motor sampling from strategy sampling** on existing checkpoints first, using the same fixed development/reserved panels and reset seeds. Compare stochastic strategy + deterministic motor against the current both-stochastic and both-deterministic modes. This isolates the confirmed migration-noise discrepancy without changing reward, PPO, or controller weights. Native CPU yaw RMSE was instantaneous jitter, not proof of sustained turning bias or the cause of strategic circling.
2. **Apply identical reserved evaluations to control and potential0.50**, both training seeds, before choosing a recipe. Keep reference, role, reset distribution, and action modes explicit; do not choose using self-play alone.
3. If pursuing another training comparison, retain a matching control and test the best balanced candidate over **more independent training seeds**. Separate any motor-sampling change from reward/normalization changes. Predefine a both-role criterion with TAG, full-window completion, own OOB/fall, and transfer diagnostics; the current sweep does not meet reliability requirements.
4. Broader starts should be a separately measured curriculum/reset experiment, not assumed beneficial from geometric variety. Investigate approach versus contact completion with trajectories, minimum distance and terminal causes against **moving references**, including each reference separately.

No Slurm submission/cancellation or production-code change was performed for this analysis.
