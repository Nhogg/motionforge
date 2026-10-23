# Model Architecture: Two-G1 Humanoid Tag

Notes for drawing the architecture figure. All numbers come from the code
(`src/mofo/tasks/tag.py`, `src/mofo/selfplay.py`, `src/mofo/opponent.py`,
`src/mofo/league.py`, `scripts/mofo_train.py`) and from the stock mjlab G1 configs it builds on.
The observation and action sizes were checked by building the env.

---

## 1. Overview (one-sentence caption)

A whole-body humanoid policy (Unitree G1, 29 joints) is trained in **two stages**. **Stage 1**
is the stock mjlab velocity-tracking locomotion policy. **Stage 2** warm-starts from it,
swaps the 3-D velocity command input for 6-D **tag features** (opponent state, role, wall
distance), and fine-tunes it with PPO against a **league** of opponents: a scripted stage-1
chaser/fleer, frozen snapshots of the learner (sampled with PFSP), and **symmetric
self-play**, where the learner controls both robots.

The figure works best as three panels:

- **(a) Policy / value networks:** actor-critic MLPs with their inputs and outputs.
- **(b) Warm start:** how the stage-1 weights map onto the stage-2 input layer.
- **(c) Training system:** env partition, opponent league, and per-seat rewards feeding PPO.

---

## 2. Panel (a): Actor-critic networks

### Actor (policy) π_θ

| Stage | Detail |
|---|---|
| Input | o_actor ∈ ℝ¹⁰², noisy during training, clean at eval |
| Normalization | Empirical running mean/std normalizer (per dimension) |
| Trunk | MLP 102 → 512 → 256 → 128, ELU activations |
| Head | Linear 128 → 29 = action mean μ |
| Distribution | Gaussian N(μ, σ²), σ ∈ ℝ²⁹ is a learned, state-independent parameter (init 1.0) |
| Output | a ∈ ℝ²⁹ (deterministic μ at eval and for frozen opponents) |

### Critic (value) V_φ: asymmetric / privileged

| Stage | Detail |
|---|---|
| Input | o_critic ∈ ℝ¹¹⁵, no noise |
| Normalization | Empirical running mean/std normalizer |
| Trunk | MLP 115 → 512 → 256 → 128, ELU |
| Head | Linear 128 → 1 = V(s) |

Actor and critic are **separate networks**; no weights are shared.

### Observation vector

| Term | Dim | Actor | Critic | Notes |
|---|---:|:-:|:-:|---|
| base linear velocity (IMU) | 3 | ✓ (noisy) | ✓ | |
| base angular velocity (IMU) | 3 | ✓ (noisy) | ✓ | |
| projected gravity | 3 | ✓ (noisy) | ✓ | |
| joint positions (rel. default) | 29 | ✓ (biased + noisy) | ✓ (true) | critic sees the unbiased value |
| joint velocities | 29 | ✓ (noisy) | ✓ | |
| last action | 29 | ✓ | ✓ | |
| foot height | 2 | | ✓ | privileged |
| foot air time | 2 | | ✓ | privileged |
| foot contact | 2 | | ✓ | privileged |
| foot contact forces | 6 | | ✓ | privileged |
| **tag features** | **6** | ✓ | ✓ | see below |
| episode progress t/T | 1 | | ✓ | privileged; the timeout decides the game |
| **Total** | | **102** | **115** | |

**Tag features (6-D), in the robot's yaw-aligned frame:**
1–2. opponent relative position (x, y)
3–4. opponent relative velocity (x, y)
5. role: +1 pursuer, −1 evader (resampled 50/50 each episode)
6. own distance to the nearest arena wall

The proprioceptive block (96-D) is identical to stage 1. Only the last block changes:
stage 1 had the **3-D velocity command** (v_x, v_y, ω_z) where stage 2 has the **6-D tag features**.

### Action → robot

```
a ∈ ℝ²⁹  ──►  q* = q_default + s ⊙ a     (s_j = 0.25 · τ_max,j / K_p,j)
         ──►  joint PD position controllers (MuJoCo, 200 Hz physics)
policy rate 50 Hz (decimation 4, Δt = 0.02 s)
```

### ASCII: panel (a)

```
                          ┌─────────────────────── ACTOR π_θ ───────────────────────┐
 Proprioception (96) ──┐  │                                                          │
   ω, v, g, q, q̇, a_t-1 │  │  ┌──────────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌────────┐   │   μ ∈ ℝ²⁹
                        ├──┼─►│ RunNorm  ├─►│ 512 ├─►│ 256 ├─►│ 128 ├─►│ Linear ├───┼──► N(μ, σ²) ──► a_t
 Tag features (6) ──────┘  │  │  (102)   │  │ ELU │  │ ELU │  │ ELU │  │  → 29  │   │    σ: learned
   Δp_xy, Δv_xy,           │  └──────────┘  └─────┘  └─────┘  └─────┘  └────────┘   │       │
   role, d_wall            └──────────────────────────────────────────────────────────┘       │
                                                                                              ▼
                                                                          q* = q₀ + s ⊙ a_t
                                                                          PD control @200 Hz
                                                                          (policy @50 Hz)

                          ┌─────────────────────── CRITIC V_φ ──────────────────────┐
 Proprio (96, clean) ──┐  │                                                          │
 Foot state (12) ──────┤  │  ┌──────────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌────────┐   │
   height, air time,   ├──┼─►│ RunNorm  ├─►│ 512 ├─►│ 256 ├─►│ 128 ├─►│ Linear ├───┼──► V(s)
   contact, forces     │  │  │  (115)   │  │ ELU │  │ ELU │  │ ELU │  │  → 1   │   │
 Tag features (6) ─────┤  │  └──────────┘  └─────┘  └─────┘  └─────┘  └────────┘   │
 Episode progress (1) ─┘  └──────────────────────────────────────────────────────────┘
```

---

## 3. Panel (b): Warm start from stage 1

Stage 2 copies every stage-1 weight except the first layer's input columns, which are
remapped term by term:

- **Shared proprio columns (96):** copied, together with their normalizer statistics.
- **Velocity-command columns (3):** dropped. Their contribution under a *zero* command is
  folded into the first-layer bias: b' = b + W_cmd · (−μ_cmd / (σ_cmd + ε)).
- **New tag-feature columns (6), plus progress (1) in the critic:** initialized to zero.
- **Hidden layers, output layer, action std:** copied unchanged.

So at iteration 0 the tag policy is **exactly the stage-1 policy commanded to stand still**.
The code asserts max |a_tag − a_stage1(cmd=0)| < 1e-3.

```
  Stage 1 first layer  W₁ ∈ ℝ^{512×99}            Stage 2 first layer  W₁' ∈ ℝ^{512×102}
  ┌──────────────────────┬─────────┐              ┌──────────────────────┬──────────────┐
  │   proprio (96 cols)  │ cmd (3) │              │   proprio (96 cols)  │ tag (6 cols) │
  └──────────┬───────────┴────┬────┘              └──────────▲───────────┴──────▲───────┘
             │                │                              │                  │
             └──── copy ──────┼──────────────────────────────┘             zeros (new)
                              │
                              └──► fold W_cmd·n(0) into bias b₁'  (zero-command bias)

  Layers 2..4, output head, σ:  copied unchanged
  Optimizer: fresh Adam, lr 1e-4 (stage 1 used 1e-3)
```

---

## 4. Panel (c): Training system

### Components

1. **Simulator:** N = 4096 parallel MuJoCo-Warp worlds. Each holds two G1s
   (`robot`, `opponent`) in a square arena (half-extent 4 m), with 20 s episodes.
2. **Env partition** (contiguous slices of the N envs):
   - **Self-play envs** (fraction `selfplay_frac`): the learner controls **both** robots.
     Each env contributes **two** training samples (one per seat), so PPO sees N + n_sp
     trajectories.
   - **League groups** (`groups` slices): the learner controls `robot`. The opponent seat
     is driven by one frozen opponent per group, redrawn every 1200 env steps.
3. **Opponent league (PFSP):** the pool contains
   - the **scripted opponent**: the frozen stage-1 velocity policy fed a hand-coded
     command (toward the other robot as pursuer; away from it and bent back toward the
     center near the walls as evader). It is never evicted.
   - **learner snapshots**, frozen every `snapshot_every` iterations (pool max 32, oldest evicted first);
   - optional preloaded **historical** checkpoints.

   Sampling weight ∝ (1 − p_win)^2, where p_win is an EMA of the learner's win rate
   against that member. Every member's probability is floored at 0.02.
4. **Per-seat rewards:** mjlab's reward terms are split by seat. The zero-sum terms (tag
   outcome, distance shaping) enter the opponent seat **negated**. Locomotion regularizers
   are mirrored, so each seat pays its own.
5. **PPO** (rsl-rl): 24 steps per env per iteration, 5 epochs × 4 mini-batches, γ = 0.99,
   λ = 0.95, clip 0.2, entropy 0.01, adaptive-KL learning rate (target KL 0.01).

### Reward (learner seat)

| Term | Form |
|---|---|
| Tag outcome (sparse, terminal) | ±10: pursuer wins on tag, evader wins on timeout; falling or leaving the arena loses |
| Distance shaping (optional) | F = γΦ(s') − Φ(s), Φ = −role · ‖p_robot − p_opp‖_xy, Φ(terminal) = 0 (potential-based, unbiased) |
| Gait terms (optional) | stock pose / foot-slip / swing-height terms, using the robot's *own measured velocity* as the command |
| Regularizers | stock mjlab G1 terms (action rate, foot clearance, soft landing, upright, …) |

### ASCII: panel (c)

```
 ┌──────────────────────────────── N = 4096 parallel MuJoCo worlds ────────────────────────────────┐
 │                                                                                                  │
 │   ┌──── Self-play envs (n_sp) ────┐   ┌──── League group 1 ────┐  …  ┌──── League group G ────┐  │
 │   │  robot ◄── π_θ                │   │  robot    ◄── π_θ      │     │  robot    ◄── π_θ      │  │
 │   │  opponent ◄── π_θ (same net)  │   │  opponent ◄── Opp_1    │     │  opponent ◄── Opp_G    │  │
 │   └───────────┬───────────────────┘   └──────┬─────────▲───────┘     └──────┬─────────▲───────┘  │
 └───────────────┼──────────────────────────────┼─────────┼────────────────────┼─────────┼──────────┘
                 │ obs, r_robot, r_opp          │ obs,    │ opponent           │         │
                 │ (2 samples / env)            │ r_robot │ policy             │         │
                 ▼                              ▼         │                    ▼         │
         ┌────────────────────────────────────────────┐   │    ┌─────────────────────────┴───────┐
         │  Per-seat reward split                      │   │    │        OPPONENT LEAGUE          │
         │  r_opp = mirrored regularizers − zero-sum   │   │    │  ┌──────────────────────────┐   │
         └──────────────────────┬─────────────────────┘   └────┤  │ Scripted: stage-1 policy │   │
                                ▼                               │  │  + chase/flee command    │   │
         ┌────────────────────────────────────────────┐        │  ├──────────────────────────┤   │
         │  PPO (rsl-rl)  on N + n_sp trajectories     │        │  │ Snapshots of π_θ (≤32)   │   │
         │  actor π_θ  /  critic V_φ  (panel a)        │        │  ├──────────────────────────┤   │
         └──────────────────────┬─────────────────────┘        │  │ Historical ckpts (opt.)  │   │
                                │  every K iterations: freeze π_θ ──►└──────────────────────────┘   │
                                └───────────────────────────────►│  PFSP: P ∝ (1 − p̂_win)²        │
                                                                 │  p̂_win: EMA of match outcomes │◄── episode outcomes
                                                                 └─────────────────────────────────┘
```

---

## 5. Stage 1 (optional inset)

The stock mjlab G1 flat-ground velocity-tracking task. Same network shapes as stage 2:
the actor input is 99-D (96 proprio + 3 command) and the critic input is 111-D (99 + 12 foot state). The policy
is trained to track randomly sampled (v_x, v_y, ω_z) commands, with lr 1e-3. In stage 2 it
plays two roles: it is the **initialization** for the learner, and the **low-level
controller** of the scripted opponent, which only supplies its velocity command.

```
 Stage 1: velocity tracking                      Stage 2: tag
 ┌──────────────────────────┐   warm start   ┌──────────────────────────┐
 │ π₁(proprio, v_cmd)       ├───────────────►│ π_θ(proprio, tag feats)  │  learner
 │ trained on random v_cmd  │                └──────────────────────────┘
 └────────────┬─────────────┘
              │ frozen                        ┌──────────────────────────┐
              └──────────────────────────────►│ π₁(proprio, v_script)    │  scripted opponent
                                              │ v_script = chase / flee  │
                                              └──────────────────────────┘
```

---

## 6. Suggested figure styling

- Use one color per role: **pursuer red, evader light blue**, matching the video tints in
  `config/mofo_train.yaml`.
- Draw trainable modules with solid borders and frozen ones (snapshots, scripted stage-1
  policy) with dashed borders or a ❄ icon.
- Highlight what is new relative to stage 1: the tag-feature input block, the zero-init
  columns, and the league loop.
- Label tensor sizes on the edges (102, 115, 29, 1) rather than inside the boxes.
