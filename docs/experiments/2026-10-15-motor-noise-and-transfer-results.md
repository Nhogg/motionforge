# Motor-noise and transfer results — 2026-10-15

## Bottom line

**Removing frozen-motor sampling during training did not solve pursuit and made two paired seeds substantially worse.** Removing it only during evaluation has small, inconsistent effects. Actuator noise contributes tracking jitter, but is not a sufficient explanation for persistent strategic retreat/circling.

Completed matched-reference validation also does not identify a reliable both-role winner. GAE1 seed1 is promising; seed2 and other opponents expose large failures. Small initialization and global normalization can improve pursuit while leaving the evader easy to tag.

The corrected anchored plots now contain real, role-separated variation rather than the old forced 0.5 cancellation. Their meaning remains **online performance relative to one frozen reference**, not absolute skill or validated held-out success probability.

## Data and scoring

- Recompiled `logs/hierarchical/diagnostics/overnight/comparison.json`: all 40 standard panels complete, all eight motor mixed-mode panels complete, all 12 candidate/control reserved panels complete.
- Motor experiment: array4385, four training seeds0–3, paired control versus deterministic-frozen-motor training. Reward alpha0.25, frozen locomotion, 100 iterations. Initial learner-state SHA256s match exactly within all four pairs.
- Final fixed-historical cells combine checkpoints005/020 and separations1.5/2m: **64 episodes per training seed, role and action mode**. This aggregation is descriptive; reference/distance-separated rows remain in the JSON. Evaluation episodes are not independent training replicates.
- D: deterministic strategy and motor. S: stochastic strategy and motor. M: stochastic strategy, deterministic motor. Every learned checkpoint is tested under all three modes.
- P = valid TAGs / all episodes. E = physically survived, tag-free `TIME_LIMIT_TRUNCATION` / all episodes. Own or opponent OOB/fall is not E completion. Neither metric is the game's terminal payoff or an uncensored win rate.

## 4385: paired training result

Each cell is **(P TAG count, E full-window count)**, each out of64.

| Training variant | Seed | D evaluation | S evaluation | M evaluation |
|---|---:|---|---|---|
| Control | 0 | (42,0) | (56,1) | (48,0) |
| Motor deterministic | 0 | (0,0) | (0,1) | (0,0) |
| Control | 1 | (16,64) | (25,51) | (28,53) |
| Motor deterministic | 1 | (16,48) | (19,53) | (16,55) |
| Control | 2 | (48,64) | (25,64) | (26,64) |
| Motor deterministic | 2 | (0,64) | (19,64) | (21,64) |
| Control | 3 | (16,64) | (12,60) | (10,62) |
| Motor deterministic | 3 | (16,64) | (14,60) | (11,60) |

Interpretation:

- Motor-deterministic training loses seed0 pursuit in **all** evaluation modes. Seed2 loses all deterministic pursuit and remains weaker under S/M. Seed1 is not rescued; seed3 changes are small.
- Control checkpoints evaluated M instead of S change P TAGs by **[-8,+3,+1,-2]/64** across seeds0–3. There is no consistent evaluation-only benefit.
- Seed0 evader collapses in both training arms: deterministic and M evaluation get tagged64/64; S gets tagged63/64. That is opponent contact, not an OOB classification artifact.
- No falls are recorded in these final fixed-historical motor panels. Own pursuer OOB occurs in motor-deterministic seed0: **11/64 D, 7/64 S, 6/64 M**. The other displayed own-role physical failures are zero. Thus motor-off seed0's failure is mostly no-tag truncation/retreat, with additional OOB, not universal falls.
- The sampling switch is **global to both agents**. Training/evaluation changes motor noise for learner and opponent, not only learner actuators. Claims of a learner-only noise intervention would be incorrect.

### Initial versus final approach from 2m

Mean distance closed = initial root distance minus final root distance, in metres, over the two historical references (32 episodes per cell). Each entry is **checkpoint0 → checkpoint100**. Negative means finishing farther away; it is not itself a TAG definition.

| Training variant | Seed | D distance closed | S distance closed | M distance closed |
|---|---:|---|---|---|
| Control | 0 | +1.301 → +0.847 | +0.986 → +1.195 | +1.047 → +0.932 |
| Motor deterministic | 0 | +1.299 → −2.133 | +1.008 → −2.374 | +1.035 → −2.391 |
| Control | 1 | +0.683 → −0.473 | +0.391 → −0.698 | +0.438 → −0.545 |
| Motor deterministic | 1 | +0.691 → −0.545 | +0.387 → −0.966 | +0.446 → −0.987 |
| Control | 2 | +0.301 → +1.568 | −0.165 → +0.931 | −0.519 → +0.918 |
| Motor deterministic | 2 | +0.291 → +0.575 | −0.154 → +1.083 | −0.514 → +1.313 |
| Control | 3 | −0.436 → −1.066 | −0.472 → −1.197 | −0.378 → −1.156 |
| Motor deterministic | 3 | −0.436 → −1.087 | −0.491 → −1.212 | −0.383 → −1.201 |

Seed0 motor-off begins with reasonable approach but learns sustained retreat. This is not simply an incapable initial locomotion controller. Seed2 motor-off D finishes at mean minimum distance1.378m, versus0.416m for its control, whereas its M minimum improves to0.670m: strategy sampling can help that checkpoint approach, but does not produce robust deterministic pursuit.

Seeds1 and3 have zero 2m D TAGs in both arms. Seed0 control's final2m TAGs are16/32 D,26/32 S,20/32 M; seed2 control gets32/32 D but10/32 S and11/32 M. Action mode and initial distance must remain explicit.

Small initial-evaluation differences occur even with identical initial-state hashes and matching nominal test seeds. Do not interpret one-episode differences as an exact bitwise paired causal effect; retain evaluation/physics variability as a limitation.

## Candidate transfer against matching controls

Arrays4375/4384 evaluate checkpoint100 at2m, reset seeds5000–5015, against reserved baseline950 and ablation500. Each cell is **(P TAGs,E full-window)**, each out of32. The references are outside the configured historical league, but have been development-used previously; they are not pristine untouched test data.

| Training source / seed | Face-to-face D | Face-to-face S | ±90 headings D | ±90 headings S |
|---|---|---|---|---|
| 4324 control /1 | (32,16) | (32,25) | (15,23) | (14,21) |
| 4324 potential0.50 /1 | (32,16) | (31,17) | (13,20) | (12,17) |
| 4324 GAE1 /1 | (32,32) | (31,32) | (16,28) | (14,26) |
| 4324 control /2 | (0,32) | (2,32) | (5,22) | (5,22) |
| 4324 potential0.50 /2 | (0,32) | (0,32) | (6,22) | (5,23) |
| 4324 GAE1 /2 | (16,16) | (18,18) | (7,22) | (6,23) |
| 4340 init0.01 /1 | (32,0) | (30,6) | (13,16) | (15,18) |
| 4340 init0.01 /2 | (32,11) | (17,15) | (10,10) | (11,16) |
| 4349 control /1 | (32,16) | (32,26) | (15,23) | (14,23) |
| 4349 global normalization /1 | (32,0) | (29,6) | (12,19) | (13,18) |
| 4349 control /2 | (0,32) | (2,32) | (5,22) | (5,26) |
| 4349 global normalization /2 | (2,32) | (8,32) | (5,22) | (4,23) |

Findings:

- Stronger potential does **not** show its apparent development advantage on the reserved panel. Seed1 evasion is worse than control under S; seed2 pursuit remains0/32 face-to-face D/S. Do not scale this recipe as a demonstrated winner.
- GAE1 seed1 is the most balanced individual candidate here. Seed2 improves face-to-face pursuit but loses evasion (16/32 D versus control32/32). It also had zero deterministic development pursuit. This is substantial opponent dependence, not seed-robust mastery.
- Small initialization helps seed2 reserved pursuit but severely compromises evasion. Seed1 deterministic evader gets tagged32/32. Global normalization has the same seed1 face-to-face evasion failure; its seed2 transfer gain is modest.
- Heading-randomized P counts remain low for every seed2 candidate:5–10/32 D,4–11/32 S. Better face-to-face scores have not established robust directional control.
- Init0.01's closest baseline is the4324 default control, not a contemporaneous full-rate control in4340. Shared nominal settings/seeds do not remove cross-job numerical variability.

## Are the corrected headline plots meaningful?

Yes, as **anchored online diagnostics**; no, as absolute publication-ready skill/difficulty measurements.

Checked actual corrected scalar data for all32 companion runs (3,200 records), plus the rendered GAE1 seed1 two-role figure:

- No forced0.5 combined curve remains. Examples: GAE1 seed1 final relative proxies are P0.895/E0.989; seed2 P0.102/E0.979. Heading±90 seed2 finishes P0.071/E0.664. Those are distinct role trajectories, not cancelling role Elo values.
- Difficulty is a fixed reference's empirical failure probability against the sampled historical opponent. For heading0/stochastic motor, P difficulty levels are **11/32,17/32,19/32,21/32**; E levels are **30/32,31/32,30/32,25/32**. There are only four P levels and three distinct E levels. Flat stretches or discrete jumps are expected, honest measurements—not a reason to add artificial variation.
- Calibration coverage is82% of training episodes in these runs; unknown current/scripted/stationary items leave difficulty gaps. The green estimate can retain its last calibrated value across such gaps, with no update. Do not present coverage as100%.
- The evader anchor is weak against these historical opponents (only1–7 full-window completions per32). Consequently high E relative skill means outperforming **that anchor**, not98% evasion against reserved opponents. Some high-E online estimates coexist with poor transfer. The replay is retrospective and the anchor/panel are development-used.
- Reward, online ability and opponent difficulty can now be plotted without a mathematical cancellation. But this experiment has a fixed opponent sampler: a varying red line indicates different sampled items, not proven automatic curriculum progression or increasing environmental challenge.

For a headline figure, show paired fixed-reference evaluation curves/seed uncertainty alongside these diagnostic plots. Keep role-separated panels, shaped reward on its own axis, coverage and reference definition visible, and do not relabel sigmoid ability as an empirical TAG probability.

## What should be tested next?

1. **Action-space/closed-loop interventions are warranted; globally disabling motor noise is not the leading fix.** At saved checkpoints, measure effects of yaw suppression/caps, forward/backward component removal and scripted bearing-feedback replacement using matched opponent, role, start and action mode. Record commanded and achieved body velocity/yaw, distance, contact and causes. Treat these as diagnostic counterfactuals, not learned-policy success claims.
2. Backward walking is not intrinsically invalid: the accepted motor controller can track it, and a pursuer can legitimately approach backward. Test whether actions close distance and complete contact; do not ban reverse solely because a render looks odd.
3. If training a yaw-limited or alternative directional interface, isolate it from reward/optimizer changes and preserve correct action scaling/log-probability semantics. Post-hoc clipping without accounting for the policy distribution would confound the test.
4. Any next promising recipe needs more independent training seeds, both-role transfer checks, and broader relative starts. Use GAE1 seed1 as a diagnostic exemplar, not a selected universal recipe. Keep TAG payoff/falls/OOB distinct, and do not optimize only pursuer success while hiding evader collapse.

No jobs submitted/cancelled or production code changed for this analysis. Source reports were refreshed read-only from completed raw evaluations; the matching-control and mixed-mode raw files remain untouched.
