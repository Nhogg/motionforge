# 17 h sprint: feasibility and failure audit

Written 2026-10-23, before any sprint training. Repository: /home/nhogg/mot (branch L_s1, uncommitted changes).
Baseline numbers: `results/17h_baseline/` (tool `logs/tools/eval17.py`; self-play unless noted, 2 x 4096 games,
reset seeds 0 and 1, deterministic actions).

Two games appear below. **Original** = `tag_scratch_lim` (8 x 8 m arena, spawns ~2 m apart facing each other,
20 s timeout that is a draw, falls/exits are draws costing -10 vs -5 for being tagged). **Balanced** =
`tag_chase` (16 x 16 m arena, spawns 2-8 m apart, 10 s timeout that the evader wins, zero-sum clock
+-0.5/s, falls/exits lose). The sprint baseline (arm A) is the balanced game with fixed 50/50 roles
(`tag_chase_adapt`, `league.evader_rate_step=0`; runs RU000_s0/s1).

## A. Locomotion

1. **World-frame translation reward?** No. The only rewards are the game outcome (+-5), the clock (+-0.5/s),
   a negligible distance-change shaping term (0.001 per m) and a joint-limit penalty (-0.1). Nothing rewards
   sliding directly, but nothing prevents it either, and an immediate tag makes a dive free: the pursuer's
   reward arrives at contact, before the fall it causes. Measured: the original game's pursuers move at
   3.8-4.0 m/s with stance-foot slip of 1.8-2.6 m/s; tag policies fall in 95-98% of goal-task episodes
   (`logs/plots/paper/goal_transfer.json`).
2. **Foot slip, torque, torso posture, falls penalized?** Falls: yes (episode ends; a loss in the balanced
   game, -10 in the original). Slip, torque and torso posture: no, in both games (`rules.regularizers=false`;
   only `dof_pos_limits` stays).
3. **Foot contact and foot velocity available?** Yes. `feet_ground_contact` (mjlab ContactSensor on each
   ankle-roll subtree vs the terrain: `found`, `force`, air time) plus foot-site velocities
   (`site_lin_vel_w`, sites `left_foot`/`right_foot`); self-play mirrors the sensor onto the opponent
   (`opp_feet_ground_contact`). The logged metric `Episode_Metrics/foot_slip_speed` already uses them.
4. **Penalties physically meaningful and bounded?** The stock mjlab `feet_slip` is sum(contact x |v_xy|^2),
   gated on a velocity command (in `tag_walk` the robot's own measured velocity). Squared speeds are bounded in
   practice by the termination on falls. The stock `upright`/`pose` terms are *positive* and pay for standing:
   in 4 of 7 overnight `tag_walk` seeds the +0.49/episode pose reward outweighed the game reward (~0.04) and
   the policy collapsed to standing still (std 0.05, every game a timeout).
5. **Action space suitable?** 29-D joint position targets through the stock PD actuators (the action space mjlab's
   G1 velocity task walks with), clamped to the joint ranges (`limit_targets`). Caveat: exploration noise past
   the clamp is free, so the action std drifts upward (1.4-1.8 rad in the no-shaping runs at ~43k iterations).

## B. Game balance

1. **Identical capabilities?** Yes: the same G1 model, actuators and (in self-play) the same policy in both seats.
2. **Capture distance?** Base xy distance < 0.5 m, effective immediately (`tag_confirm_s=0`).
3. **Arena?** Bounded square (8 or 16 m); leaving it ends the game (a draw in the original, a loss in the
   balanced game).
4. **Can an equal-speed evader escape?** In the original game, no: spawns ~2 m apart facing each other close
   faster than an evader can turn and run; evader survival 0% at the final checkpoint of all 3 seeds
   (median game 0.9-1.0 s). In the balanced game it can: RU000 evaders win 40-43% of self-play games
   (median game 1.7-6.1 s). See the structural check below.
5. **Reward asymmetry favoring pursuers?** In the original game, yes: the timeout was a draw and falls/exits
   cost more than being tagged, so the evader had nothing to win. The balanced game is zero-sum (outcome,
   clock and shaping all negated for the other seat).
6. **Termination and truncation?** Correct: tag, falls and exits are terminal; the timeout is terminal too
   (`is_finite_horizon=True`: it is a game outcome, not a truncation). The actor has no clock input; the critic
   has episode progress. Non-finite observations reset the world as a draw (NaN guard, disclosed).

## C. Training

1. **Does each role get learning signal?** In the balanced game mostly yes: evader win rates in training are
   4-46% across seeds (RU000_s1 is the weak one: 4%). In the original game the evader never wins.
2. **Is the evader always training against opponents it cannot beat?** No: 80% of envs are league games
   against past snapshots (PFSP over up to 32 members, near-uniform in practice) and 20% self-play.
3. **Enough weaker and comparable opponents?** Weaker snapshots, yes. The league's "scripted" member is
   degenerate: with no stage-1 checkpoint it outputs zero actions and falls at once (`vs_zero`: 94-99% of games
   end by its fall), i.e. free wins that teach nothing. Learned evaders are brittle against an unfamiliar
   opponent: against the scripted stage-1 chaser, RU000_s1's evader falls in 70% of games.
4. **A checkpoint with acceptable locomotion?** No learned tag checkpoint walks acceptably (side-view videos:
   sprint-and-dive or sliding shuffle). The stage-1 velocity-tracking policy
   (`/home/nhogg/motionforge/logs/mofo/stage1_snapshot/model_17800.pt`) walks, and the existing
   `opponent.ScriptedPolicy` turns it into a scripted chase/flee opponent.
5. **Reuse checkpoint parameters for curriculum training?** Technically yes (`cmd.init` warm start), but every
   run is from scratch by the project's constraint, so no parameters are reused. The stage-1 policy is used
   only as an opponent (C3).

## Structural balance check (evaluation-time overrides only; training configs unchanged)

Pursuer win % in self-play when only the spawn separation or the capture radius changes at evaluation time
(`results/17h_baseline/struct_*.json`; ~8192 games each, 95% Wilson intervals about +-1 point):

| Checkpoint | d 2-3 m | d 4-5 m | d 6-8 m | default (2-8 m) | radius 0.3 m | radius 0.8 m |
|---|---|---|---|---|---|---|
| RU000_s0 (balanced) | 55 | 58 | 56 | 57 | 59 | 54 |
| RU000_s1 (balanced) | 54 | 65 | 60 | 60 | 62 | 57 |
| L_s1 (original) | 49 (d 1-2) | 47 (d 3-5) | - | 47 | - | - |

In the original game the evader wins nothing at any separation (timeouts 0%; its "wins" there are the
pursuer's falls, which count as draws for scoring, so both seats sit near 48%): the rules, not the spawn
distance, decide it. In the balanced game the pursuer's edge (54-65%) barely moves with separation or capture
radius. What decides the evader's losses is locomotion: evaders end 7-23% of games by falling and 8-24% by
leaving the arena, against 1-3% falls for pursuers. Against the scripted stage-1 chaser, which never catches
them, RU000_s1's evaders still lose 94% of games, by falling (70%) and leaving (18%).

**Conclusion:** pursuer domination in the balanced game is not a geometry effect, so a spawn-distance
curriculum is not justified. The evader's losses are mostly locomotion failures, which the locomotion
intervention targets.

## Interventions

| Intervention | Existing support | Implementation risk | Expected benefit | Decision |
|---|---|---|---|---|
| Foot slip penalty (stance only) | Yes: contact sensor + foot-site velocity; stock `feet_slip` exists but is command-gated | Low: ~10 lines (`tag.stance_slip`, ungated, swing feet free) | Targets the shuffle (stance slip 0.7-1.9 m/s in the baseline) | **Implement** (`rules.stance_slip_weight`) |
| Torso stability | Yes: projected gravity | Low: ~5 lines (`tag.tilt_excess`, 20 deg free zone) | Targets the 17-37 deg lean of the no-shaping runs; leaves a running lean free | **Implement** (`rules.tilt_weight`, `rules.tilt_free`) |
| Control effort (action rate) | Yes: stock `action_rate_l2` | Medium: at stock weight (-0.1) it priced out exploration and the policy stood still (run B) | Unclear | **Skip** |
| Contact-based gait constraint / stepping reward | Yes: stock `feet_air_time`, `pose` | High: positive gait rewards caused standing collapse in 4 of 7 seeds; the sprint plan forbids them | - | **Skip** |
| Spawn-distance curriculum | Partly: `reset.d_min/d_max` exist, no schedule | Medium: needs a per-iteration hook on the reset event | Little: separation barely changes the balance (structural check) | **Skip** |
| Opponent difficulty curriculum | Yes: league PFSP + scripted member + `scripted_weight_start/end/decay_iters`; trainer option `opponent=<stage-1 ckpt>` | Low: configuration only | Replaces free wins against a falling dummy with a competent, beatable walking opponent early in training | **Implement** (arm C: `opponent=` stage-1, scripted weight x4 -> x1 over 10k iterations) |
| Pretrained locomotion prior | Stage-1 velocity policy exists; no motion dataset | High, and warm starting is excluded by the project | - | **Skip** (used only as an opponent) |

**Gate:** none of the chosen interventions changes the simulator or needs a dataset.
