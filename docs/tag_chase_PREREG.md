# tag_chase preregistration (written 2026-10-23 00:20 CDT, before any results)

Runs (launched 00:10 CDT, 20,000 iterations each, from scratch):
- A = `+experiment=tag_chase`, seeds 0, 1 -> logs/mofo/tag/tag_chase/A_s{0,1} (jobs 9777, 9778)
- B = `+experiment=tag_chase_reg` (A + stock locomotion regularizers), seeds 0, 1 -> logs/mofo/tag/tag_chase_reg/B_s{0,1} (jobs 9779, 9780)

Changes vs tag_scratch_lim (the diagnosis: the evader could not win, so it learned to stand still):
16 x 16 m arena; spawns 2-8 m apart (facing); 10 s timeout that the evader wins; zero-sum clock
reward (evader +0.5/s, pursuer -0.5/s; new `rules.clock_reward`); falls and exits lose the game;
entropy_coef 0.001. Also in the code since L_s1_repro2: the non-finite-observation reset guard.

Evaluation: final checkpoint model_19999.pt, self-play (the checkpoint plays both roles),
4096 games, deterministic actions, reset seed 0 (the same spawns for every run).

Pass, per run, all of:
1. evader survival (timeout) rate between 20% and 80%;
2. median game length >= 3 s;
3. pursuer falls + exits < 10% of games;
4. a side-view video shows upright running with visible fleeing and turning (judged by eye; not from numbers alone).

Selection: prefer a variant that passes in both seeds; if both variants do, prefer B. All four runs are reported
either way, including failures.

## Outcome of A/B (stopped by the user, 2026-10-23 ~01:15 CDT; not evaluated against the criteria above)
- B_s0/B_s1 (jobs 9779/9780) stopped at ~2400 iterations: the action std had collapsed to 0.05 (action_rate_l2
  priced exploration out), every game was a timeout, and the robots stood still (user watched the videos).
- A_s0/A_s1 (9777/9778) stopped at ~3550 iterations while healthy (pursuer won ~83% of self-play games, std ~0.9),
  to free GPUs for the experiment below.

# tag_chase_adapt preregistration (written 2026-10-23 ~01:25 CDT, before any results)

Question: does adapting the learner's role draw toward a 50% evader win rate give a more balanced, more
interesting chase than fixed 50/50 roles, and does the step size matter?

Runs (from scratch, 20,000 iterations, launched ~01:20 CDT; all `+experiment=tag_chase_adapt`, i.e. tag_chase
plus league.selfplay_frac 0.2):
- R050_s{0,1}: league.evader_rate_step 0.05 (jobs 9785, 9786)
- R030_s{0,1}: 0.03 (9787, 9788)
- R025_s{0,1}: 0.025 (9789, 9790)
- R000_s{0,1}: 0 = control, fixed 50/50 roles at the same selfplay_frac (9791, 9792)
Bounds: p_evader in [0.05, 0.95], start 0.5, updated once per iteration from the learner-evader games finished
in it (league envs where the learner was the evader + all self-play games).

Evaluation (same as tag_chase above): final checkpoint model_19999.pt, self-play, 4096 games, deterministic,
reset seed 0. Pass criteria per run, unchanged: evader survival 20-80%; median game >= 3 s; pursuer falls + exits
< 10%; side-view video shows upright running with visible fleeing and turning.

Comparison: an adaptive arm "helps" only if both its seeds pass and the control's do not, or if both pass and
its self-play evader survival is closer to 50% in both seeds. Among passing adaptive arms, prefer the one whose
seeds are closest to 50% survival. All eight runs are reported.

Change to the W&B eval videos (presentation only; the plots, Elo and the evaluation above keep reset seed 0):
reset seed 0 + iteration, and in 6 of the 8 video worlds the opponent seat is an earlier checkpoint of the run
drawn at random (listed in the video caption).

## Open-ended continuation sweep (added 2026-10-23 ~01:50 CDT, before any results)
Same four arms, seeds 2 and 3, `cmd.iterations=1000000` (they stop at the 24 h slurm limit, roughly 100k+
iterations): U030_s{2,3}, U050_s{2,3}, U025_s{2,3}, U000_s{2,3} (jobs 9798-9805), queued on g3 behind the
current sweep. Their model_20000.pt (save interval 500; an open-ended run writes no model_19999) is evaluated
exactly as above and pooled with seeds 0-1 for the arm comparison (4 seeds per arm at 20k). Iterations past 20k
are exploratory: reported as such, not used for the preregistered comparison.

## Revision 2026-10-23 ~02:10 CDT (before any results of the runs below)
- The R sweep (seeds 0-1, 20k cap) was stopped at ~4000 iterations and the queued U sweep (seeds 2-3) cancelled,
  both by the user: every R run had learned a sliding shuffle with waist flailing (feet air time ~0.05 s,
  foot slip ~0.8-1 m/s, torso tilt 15-30 deg). Not evaluated.
- RU sweep: the same four arms (step 0.05/0.03/0.025, control 0), seeds 0-1, restarted from scratch with no
  iteration cap (24 h slurm limit): RU{050,030,025,000}_s{0,1}, jobs 9807-9814. Question (exploratory): does
  tag alone eventually grow out of the shuffle? Preregistered comparison at model_20000.pt, evaluated as above.
- Walking sweep, from scratch, no iteration cap: `+experiment=tag_walk` (gait shaping on measured velocity:
  pose 0.5, foot_slip -0.25, foot_swing_height -0.25, air_time 0.25; body_ang_vel -0.05, angular_momentum -0.02,
  self_collisions -1; upright/foot_clearance/soft_landing dropped; action_rate_l2 -0.005) and `tag_walk_x2`
  (gait and anti-flailing weights doubled). Arms: W1A (tag_walk, adaptive step 0.03), W1F (tag_walk, fixed roles),
  W2A (x2, adaptive 0.03), W2F (x2, fixed); seed 0 each (jobs 9815-9818), plus W1A_s1 (job 9819, starts when a
  g3 GPU frees). Evaluated at model_20000.pt like the above, plus a walking criterion judged before the game
  criteria: in the side-view video both robots step (alternating feet leave the ground; no sustained sliding)
  with the torso upright, and logged feet air time >= 0.2 s and foot slip <= 0.3 m/s. With one seed per arm
  (two for W1A) the walking sweep is exploratory; it is reported in full.
- 02:25 CDT: W2A_s0/W2F_s0 (tag_walk_x2) stopped at ~1000 iterations, failed the same way as B: action std
  0.87 -> 0.16, zero tags (every game a timeout), feet near-still (slip 0.02-0.04 m/s), pose reward near its
  maximum (standing at the default pose). Not evaluated. Their GPUs went to more tag_walk seeds: W1A_s1 (job 9824),
  W1F_s1 (9825), W1A_s2 (9826, queued on g3). The walking sweep is now W1A (seeds 0-2) and W1F (seeds 0-1).
- 09:20 CDT: W1A_s1/W1A_s2 stopped (~26k/~24k iterations). Both collapsed within the first 2500 iterations the
  same way as B and W2: action std 0.06-0.12, zero tags, every game a timeout, feet still (slip ~0.01 m/s).
  W1A_s0 did not collapse but its p_evader pinned at 0.05 from ~20k and its tags fell to ~0 by 30k. They are
  reported as failures; W1A_s1/s2 have model_20000.pt and are evaluated like the others. Their GPUs went to more
  fixed-role seeds, W1F_s2 (job 9840) and W1F_s3 (9841), from scratch, exploratory (they start ~7 h after s0-s1).

## Round 3, launched 2026-10-23 ~11:45 CDT (written before any results of these runs)
Overnight failure modes (RU, W1 sweeps; evaluation table in logs/plots/eval20k, figure logs/plots/paper/sweep_eval):
1. Gait-shaped runs collapse to standing (W1A_s1/s2, W1F_s2/s3; W1A_s0's pursuer stopped tagging): the positive
   pose reward (+0.49/episode, paid most for standing at the default pose) outweighed the game reward (~0.04).
2. Pursuers sprint and dive: pursuer falls + exits up to 31% of evaluation games; the tag policies fall in 95-98%
   of goal-task episodes (logs/plots/paper/goal_transfer). An immediate tag makes the dive free.
3. Without gait shaping the action std grows to 1.4-1.8 rad and the gait stays a sliding shuffle (slip 0.6-1.5
   m/s, torso tilt 17-37 deg at ~43k iterations, no trend).
4. Adaptive role sampling pinned at its bound in 5 of 6 runs: dropped; all round-3 runs use fixed 50/50 roles.
Stopped at ~11:40 CDT: RU050_s0/s1, RU030_s0/s1, RU025_s0, RU000_s1, W1A_s0, W1F_s0/s1/s2/s3. Kept running
(exploratory, past 20k): RU000_s0 (control) and RU025_s1 (the one run passing criteria 1-3 at 20k).

Arms (from scratch, `cmd.iterations=1000000`, 24 h slurm limit, seeds 0-2 each):
- TC = `tag_confirm`: RU000 + `rules.tag_confirm_s: 1.0` (a tag counts once the pursuer stays up 1 s after contact).
- TE = `tag_confirm_ent0`: TC + entropy_coef 0.
- WN = `tag_walk_nopose`: W1F (tag_walk, fixed roles) with gait pose weight 0.
- WC = `tag_walk_confirm`: WN + tag_confirm_s 1.0.
Single-change comparisons: TC vs RU000 (confirmation), TE vs TC (entropy), WN vs W1F (pose), WC vs WN (confirmation).
Evaluation: model_20000.pt, exactly as above (self-play, 4096 games, deterministic, reset seed 0), same four pass
criteria, plus for all arms the walking criterion (air time >= 0.2 s, slip <= 0.3 m/s, video check). Additionally
reported for every run: goal-task transfer (logs/tools/goal_transfer.py: first goal reached with the goal in front,
falls) and cross-play against the tag_scratch_lim seeds. An arm "works" if all 3 seeds pass criteria 1-4; arms are
ranked by the number of passing seeds, then by mean |evader survival - 50%|. All 12 runs are reported.
