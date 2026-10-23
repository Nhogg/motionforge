"""State-visitation distributions of a checkpoint: tag in self-play (it drives both robots), or
the single-G1 goal task (task=goal).

Runs each of num_envs worlds for one full episode under the run's own game config
(<logdir>/.hydra/config.yaml) from a fixed reset seed, accumulating per role (pursuer / evader,
or goal) on the GPU: the top-down base xy density, the planar base speed and velocity, the
actuator torques as a fraction of each joint's force limit, and their correlation. Writes
<out>.npz (the histograms) and <out>.png/.pdf; --replot redraws from the npz.

  uv run scripts/plot_states.py CKPT [--num-envs 4096] [--stochastic] [--out OUT.png]
  uv run scripts/plot_states.py --replot OUT.npz
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "disable")  # Before mjlab import (it defaults to egl).

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from omegaconf import OmegaConf

ROLES = ("pursuer", "evader", "goal")  # goal: the single robot of task=goal.
COLORS = {"pursuer": "#d91a1a", "evader": "#3b82c4", "goal": "#2a9d55"}  # Video tints, evader darkened for print.
CMAPS = {"pursuer": "Reds", "evader": "Blues", "goal": "Greens"}
XY_BINS, SPEED_MAX, SPEED_BINS, TAU_BINS = 160, 6.0, 120, 100
VEL_MAX, VEL_BINS = 6.0, 150  # Planar velocity histograms over [-VEL_MAX, VEL_MAX]^2 (m/s).
_LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
BODY_ORDER = ([f"left_{j}" for j in _LEG] + [f"right_{j}" for j in _LEG] + ["waist_yaw", "waist_roll", "waist_pitch"]
              + [f"left_{j}" for j in _ARM] + [f"right_{j}" for j in _ARM])  # G1 29-DoF, torque-correlation axes.


def roles(h: dict) -> list[str]:
  return [r for r in ROLES if f"{r}_steps" in h]


def load_env(logdir: str, num_envs: int, device: str | None = None, camera: dict | None = None,
             overrides: list[str] | None = None):
  """The run's env (its .hydra/config.yaml) in eval mode: no observation noise. ``camera``:
  render with the world camera, these fields overriding the run's video.camera. ``overrides``:
  OmegaConf dotlist applied to the saved config (e.g. ["rules.learner_role=pursuer"])."""
  from mofo.env import GAME_SCHEMAS, game, make_env

  cfg = OmegaConf.load(os.path.join(logdir, ".hydra", "config.yaml"))
  cfg.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if camera:
    cfg.video.camera = OmegaConf.merge(cfg.video.camera, camera)
  if overrides:
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
  for k, schema in GAME_SCHEMAS.items():  # Typed groups, as mofo.cli gives them under Hydra.
    node = OmegaConf.structured(schema)
    OmegaConf.set_readonly(node, False)
    cfg[k] = OmegaConf.merge(node, cfg[k])
  return make_env(cfg, num_envs, game(cfg), eval=True, video=bool(camera))


def rollout(env, ckpt: str, stochastic: bool = False, seed: int = 0) -> dict[str, np.ndarray]:
  """One episode per world from env.reset(seed): the same spawns for every checkpoint."""
  from mofo.opponent import load_policy
  from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

  from mofo.tasks.tag import role, tag_rules

  dev, n = env.device, env.num_envs
  sd = torch.load(ckpt, map_location=dev, weights_only=False)["actor_state_dict"]
  policy = load_policy(ckpt, dev)
  std = sd["distribution.std_param"] if stochastic else None

  half = tag_rules(env).arena_half_extent
  solo = "opponent" not in env.scene.entities  # task=goal: one robot, role "goal".
  rs = ("goal",) if solo else ("pursuer", "evader")
  ents = {k: env.scene[k] for k in (("robot",) if solo else ("robot", "opponent"))}
  frange = torch.as_tensor(env.sim.mj_model.actuator_forcerange[:, 1], device=dev, dtype=torch.float32)
  tau_max = {k: frange[e.indexing.ctrl_ids] for k, e in ents.items()}
  hist = {f"{r}_{q}": torch.zeros(*s, device=dev) for r in rs
          for q, s in (("xy", (XY_BINS, XY_BINS)), ("speed", (SPEED_BINS,)), ("tau", (TAU_BINS,)),
                       ("vel_heading", (VEL_BINS, VEL_BINS)), ("vel_world", (VEL_BINS, VEL_BINS)))}
  steps, speed_sum = {r: 0 for r in rs}, {r: 0.0 for r in rs}
  # Torque correlation: per-role sums of tau and tau tau^T over live steps (float64; Pearson r at the end).
  nj = len(tau_max["robot"])
  mom = {r: (torch.zeros(nj, dtype=torch.float64, device=dev), torch.zeros(nj, nj, dtype=torch.float64, device=dev)) for r in rs}

  def act(o):
    mu = policy(o)
    return mu + std * torch.randn_like(mu) if std is not None else mu

  obs, _ = env.reset(seed=seed)
  live = torch.ones(n, dtype=torch.bool, device=dev)  # Each world's first episode only.
  with torch.no_grad():
    while live.any():
      pursuer_is_robot = role(env) > 0
      for name, e in ents.items():
        d = e.data
        is_p = pursuer_is_robot if name == "robot" else ~pursuer_is_robot
        seats = (("goal", live),) if solo else (("pursuer", live & is_p), ("evader", live & ~is_p))
        xy = (d.root_link_pos_w - env.scene.env_origins)[:, :2]
        speed = d.root_link_lin_vel_w[:, :2].norm(dim=-1)
        tau = d.actuator_force / tau_max[name]
        vel = {"world": d.root_link_lin_vel_w[:, :2],
               "heading": quat_apply_inverse(yaw_quat(d.root_link_quat_w), d.root_link_lin_vel_w)[:, :2]}
        for r, m in seats:
          if not m.any():
            continue
          ij = ((xy[m] + half) / (2 * half) * XY_BINS).long().clamp(0, XY_BINS - 1)
          hist[f"{r}_xy"].view(-1).index_add_(0, ij[:, 0] * XY_BINS + ij[:, 1], torch.ones(len(ij), device=dev))
          for frame, v in vel.items():  # Velocities outside the window are dropped, not clamped to its edge.
            v = v[m][(v[m].abs() < VEL_MAX).all(-1)]
            ij = ((v + VEL_MAX) / (2 * VEL_MAX) * VEL_BINS).long()
            hist[f"{r}_vel_{frame}"].view(-1).index_add_(0, ij[:, 0] * VEL_BINS + ij[:, 1], torch.ones(len(ij), device=dev))
          hist[f"{r}_speed"] += torch.histc(speed[m].clamp(max=SPEED_MAX - 1e-6), SPEED_BINS, 0, SPEED_MAX)
          hist[f"{r}_tau"] += torch.histc(tau[m].clamp(-1, 1), TAU_BINS, -1, 1)
          t64 = tau[m].double()
          mom[r][0].add_(t64.sum(0))
          mom[r][1].add_(t64.T @ t64)
          steps[r] += int(m.sum())
          speed_sum[r] += speed[m].sum().item()
      a = act(obs["actor"]) if solo else torch.cat([act(obs["actor"]), act(obs["opponent"])], -1)
      obs, _, term, trunc, _ = env.step(a)
      live &= ~(term | trunc)
  out = {k: v.cpu().numpy() for k, v in hist.items()}
  for r, (s1, s2) in mom.items():
    out[f"{r}_tau_sum"], out[f"{r}_tau_outer"] = s1.cpu().numpy(), s2.cpu().numpy()
  out["joint_names"] = np.array(ents["robot"].actuator_names)
  out.update(half=half, num_envs=n, ckpt=ckpt, stochastic=stochastic, step_dt=env.step_dt,
             **{f"{r}_steps": s for r, s in steps.items()}, **{f"{r}_speed_sum": v for r, v in speed_sum.items()})
  return out


def summary(h: dict) -> dict[str, float]:
  """Scalars per role: mean planar speed, share of torque samples at the force limit, mean episode length."""
  out = {}
  for r in roles(h):
    tau = h[f"{r}_tau"]
    out[f"{r}/speed_mean"] = float(h[f"{r}_speed_sum"]) / int(h[f"{r}_steps"])
    out[f"{r}/torque_saturation"] = float(tau[0] + tau[-1]) / tau.sum()  # |tau| within 2% of the limit.
  out["episode_s"] = int(h[roles(h)[0] + "_steps"]) / int(h["num_envs"]) * float(h["step_dt"])
  return out


def _title(h: dict, what: str) -> str:
  kind = "stochastic" if bool(h["stochastic"]) else "deterministic"
  mode = "goal reaching" if roles(h) == ["goal"] else "self-play"
  return f"{what}: {mode}, {int(h['num_envs'])} worlds, one episode each ({kind} actions)\n{os.path.relpath(str(h['ckpt']))}"


def _save(fig, out: str, pdf: bool) -> str:
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  fig.savefig(out, dpi=150, bbox_inches="tight")
  if pdf:
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
  plt.close(fig)
  print(f"saved {out}")
  return out


def plot(h: dict, out: str, pdf: bool = True) -> str:
  half, rs = float(h["half"]), roles(h)
  fig = plt.figure(figsize=(12, 9.5))
  gs = fig.add_gridspec(2, 2, height_ratios=(1.25, 1))
  vmax = max(h[f"{r}_xy"].max() for r in rs)
  for i, r in enumerate(rs):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(np.ma.masked_equal(h[f"{r}_xy"].T, 0), origin="lower", extent=(-half, half, -half, half),
                   cmap="magma_r", norm=LogNorm(1, vmax))
    ax.add_patch(plt.Rectangle((-half, -half), 2 * half, 2 * half, fill=False, lw=1.5, ec="0.3"))
    ax.set_title(f"{r}: base xy visitation ({int(h[f'{r}_steps']):,} steps)", color=COLORS[r], fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
  fig.colorbar(im, ax=fig.axes[:len(rs)], fraction=0.025, pad=0.02, label="visits (log)")

  for j, (key, lo, hi, xlabel) in enumerate((("speed", 0, SPEED_MAX, "planar base speed (m/s)"),
                                             ("tau", -1, 1, "actuator torque / joint force limit (all 29 joints)"))):
    ax = fig.add_subplot(gs[1, j])
    for r in rs:
      c = h[f"{r}_{key}"]
      edges = np.linspace(lo, hi, len(c) + 1)
      ax.stairs(c / c.sum() / np.diff(edges), edges, color=COLORS[r], lw=1.8, label=r, fill=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if key == "tau":
      ax.set_yscale("log")
    ax.legend(frameon=False)
  fig.suptitle(_title(h, "State visitation"), fontsize=11)
  return _save(fig, out, pdf)


def plot_velocity(h: dict, out: str, frame: str, pdf: bool = True) -> str:
  """Planar base velocity visitation per role, as a probability density over (v_x, v_y)."""
  edges = np.linspace(-VEL_MAX, VEL_MAX, VEL_BINS + 1)
  area = np.diff(edges)[0] ** 2
  rs = roles(h)
  dens = {r: h[f"{r}_vel_{frame}"] / h[f"{r}_vel_{frame}"].sum() / area for r in rs}
  # Log scale: every world starts from the same standing pose, so the first steps of a
  # deterministic policy stack into a few single-bin spikes that would saturate a linear scale.
  norm = LogNorm(1e-3, max(d.max() for d in dens.values()))
  fig, axes = plt.subplots(1, len(rs), figsize=(5.5 * len(rs), 4.8), sharey=True, squeeze=False)
  for ax, r in zip(axes[0], rs):
    im = ax.imshow(np.ma.masked_less(dens[r].T, 1e-3), origin="lower", extent=(-VEL_MAX, VEL_MAX, -VEL_MAX, VEL_MAX),
                   cmap=CMAPS[r], norm=norm)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_title(r.capitalize(), color=COLORS[r], fontweight="bold", fontsize=14)
    fwd, lat = ("forward ", "left ") if frame == "heading" else ("", "")
    ax.set_xlabel(f"$v_x$ {fwd}(m/s)")
    ax.set_ylabel(f"$v_y$ {lat}(m/s)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability density (log)")
  fig.suptitle(_title(h, f"Planar base velocity ({frame} frame)"), fontsize=12)
  fig.tight_layout()
  return _save(fig, out, pdf)


def torque_corr(h: dict, r: str) -> np.ndarray:
  """Pearson r between joint torques over all of a role's live steps (constant joints: nan)."""
  n = int(h[f"{r}_steps"])
  mean = h[f"{r}_tau_sum"] / n
  cov = h[f"{r}_tau_outer"] / n - np.outer(mean, mean)
  sd = np.sqrt(np.clip(np.diag(cov), 0, None))
  with np.errstate(invalid="ignore", divide="ignore"):
    return cov / np.outer(sd, sd)


def plot_torque_corr(h: dict, out: str, pdf: bool = True) -> str:
  """29 x 29 Pearson correlation of actuator torques per role: +1 joints push together, -1 opposite."""
  names = [str(n).removesuffix("_joint") for n in h["joint_names"]]
  # Anatomical order (mjlab lists actuators grouped by motor type): legs, waist, arms.
  order = [names.index(n) for n in BODY_ORDER if n in names] + [i for i, n in enumerate(names) if n not in BODY_ORDER]
  names = [names[i] for i in order]
  rs = roles(h)
  fig, axes = plt.subplots(1, len(rs), figsize=(8.5 * len(rs), 8.2), squeeze=False)
  axes = axes[0]
  for ax, r in zip(axes, rs):
    im = ax.imshow(torque_corr(h, r)[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    for b in np.cumsum([6, 6, 3, 7])[:len(names)] - 0.5:  # Group boundaries.
      ax.axhline(b, color="k", lw=0.8)
      ax.axvline(b, color="k", lw=0.8)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_title(f"{r.capitalize()} ({int(h[f'{r}_steps']):,} steps)", color=COLORS[r], fontweight="bold", fontsize=13)
  fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Pearson r of joint torques")
  fig.suptitle(_title(h, "Actuator torque correlation"), fontsize=12)
  return _save(fig, out, pdf)


def plot_all(h: dict, out: str, vel_frame: str = "heading", pdf: bool = True) -> dict[str, str]:
  """Every figure for one rollout: {name: png path}."""
  stem = os.path.splitext(out)[0]
  paths = {"states": plot(h, out, pdf), f"velocity_{vel_frame}": plot_velocity(h, f"{stem}_vel_{vel_frame}.png", vel_frame, pdf)}
  if "joint_names" in h:  # Older npz files predate the torque moments.
    paths["torque_corr"] = plot_torque_corr(h, f"{stem}_torque_corr.png", pdf)
  return paths


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("ckpt", nargs="?", help="Checkpoint (model_*.pt) of a tag or task=goal run.")
  ap.add_argument("--logdir", default=None, help="Run dir holding .hydra/config.yaml (default: the checkpoint's dir).")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--stochastic", action="store_true", help="Sample actions from the policy's Gaussian.")
  ap.add_argument("--device", default=None)
  ap.add_argument("--out", default=None, help="Default: logs/plots/states_<run>_<ckpt>.png")
  ap.add_argument("--replot", default=None, help="Redraw from a saved .npz instead of rolling out.")
  ap.add_argument("--vel-frame", choices=("heading", "world"), default="heading",
                  help="Velocity figure frame: heading (+x forward, +y left of each robot) or world.")
  a = ap.parse_args()
  if a.replot:
    h, out = dict(np.load(a.replot, allow_pickle=True)), a.out or os.path.splitext(a.replot)[0] + ".png"
  else:
    h = rollout(load_env(a.logdir or os.path.dirname(os.path.abspath(a.ckpt)), a.num_envs, a.device), a.ckpt, a.stochastic)
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    out = a.out or f"logs/plots/states_{run}_{os.path.splitext(os.path.basename(a.ckpt))[0]}{'_stoch' if a.stochastic else ''}.png"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(os.path.splitext(out)[0] + ".npz", **h)
  plot_all(h, out, a.vel_frame)


if __name__ == "__main__":
  main()
