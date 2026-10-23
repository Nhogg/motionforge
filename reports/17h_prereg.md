# 17 h sprint preregistration (written 2026-10-23 ~14:45 CDT, before any sprint training)

Arms (all from scratch, 20,000 iterations, identical env count (4096 + self-play seats), network, PPO settings,
league and capture rules; the only differences are listed):
- **A, baseline** `s17_base` = the RU000 control (tag_chase_adapt, fixed 50/50 roles). Seeds 0-1: the existing
  RU000_s0/s1 runs (same config and code path; the sprint's new reward options default to off). Seed 2: new run.
- **B, locomotion** `s17_loco` = A + stance foot-slip penalty (-0.1 per s x mean over the feet of contact x
  |v_xy|^2) + torso tilt penalty (-1.0 per s x max(tilt - 0.35 rad, 0)^2); both seats. Seeds 0-2.
- **C, opponent curriculum** `s17_curr` = A with the league's scripted member driven by the stage-1 velocity
  policy (scripted chase/flee) instead of zero actions; scripted weight x4 decaying to x1 by iteration 10,000.
  Seeds 0-2 (seed 2 dropped first if GPUs are short).
- **D, combined** `s17_both` = B + C. Seeds 0-2.
Not run: spawn-distance curriculum (structural check: separation barely changes the balance), action-rate
retuning, positive gait rewards (reports/17h_feasibility.md).

Evaluation (identical for every run): model_20000.pt, the balanced game's own config (no curriculum, no
overrides), `logs/tools/eval17.py`, reset seeds 0 and 1 (2 x 4096 games), deterministic actions;
opponents: self-play, the scripted stage-1 opponent, and the A seeds' model_20000 (cross-play). Metrics per role:
win % with 95% Wilson intervals, capture time (mean, median), episode length, falls, exits, planar speed, stance
foot slip, torso tilt. Videos: side views (logs/tools/side_view.py) at reset seed 0, the same worlds for every arm.

Primary outcomes, per arm over seeds (mean and per-seed values; with 3 seeds, differences are described as
preliminary unless every seed of one arm beats every seed of the other):
1. Locomotion: stance foot slip and fall rate, by role (self-play). Improvement = lower in every seed than A's mean,
   without pursuer win % falling below 40% against A's checkpoints in cross-play (no major loss of pursuit).
2. Balance: evader win % in self-play, and evader survival time; closer to 50% is better, but a 50% rate is not
   by itself evidence of learning (also report cross-play against A).
Early stopping: non-finite values; action std below 0.1 with zero tags for 1,000 iterations (standing collapse);
fall rate above 50% after iteration 5,000 with no recovery for 2,000 iterations.
Verdict per the sprint plan: Improved / Mixed / Negative, stated in reports/17h_results.md.
