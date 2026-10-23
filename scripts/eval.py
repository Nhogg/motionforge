"""Self-play viewer: one tag checkpoint drives both robots in num_envs worlds, overlaid in one MP4.

Hydra config: config/eval.yaml. Every world sits at the origin (env_spacing=0), so one fixed
camera sees each world's game in the same place. Each frame starts from the floor (pixels no
world's robot covers) and alpha-blends every world's robot pixels, found from the camera's
segmentation, so the num_envs games read as ghosts. Also writes <out>.png (base trails) and
<out>.json (per-world outcomes).

  uv run scripts/eval.py policy=CKPT [stochastic=true] [freeze=false steps=1500] [reset=legacy] [video.camera.up=9]
"""

import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "disable")  # Before mjlab import (it defaults to egl).

import mujoco
import torch
from omegaconf import DictConfig

from mofo.cli import run
from mofo.env import GAME_SCHEMAS, game, make_env
from mofo.opponent import load_policy
from mofo.tasks.tag import outcome, role

def project(cam, pts: torch.Tensor) -> torch.Tensor:
  """Pixel (row, col) of world points [n, 3] (env 0's frame) through the world camera, off-image ones dropped."""
  c = torch.tensor([0.0, -cam.back, cam.up])
  fwd = -c / c.norm()
  right = torch.tensor([1.0, 0.0, 0.0])
  up = torch.linalg.cross(right, fwd)
  f = cam.height / 2 / math.tan(math.radians(cam.fovy or 45.0) / 2)
  p = pts.cpu().float() - c
  z = p @ fwd
  rc = torch.stack([cam.height / 2 - f * (p @ up) / z, cam.width / 2 + f * (p @ right) / z], -1).round().long()
  return rc[((rc >= 0) & (rc < torch.tensor([cam.height, cam.width]))).all(-1)]


def arena_outline(cam, half: float, n: int = 400) -> torch.Tensor:
  """Pixel (row, col) of the arena border on the floor, projected through the world camera."""
  s = torch.linspace(-half, half, n)
  edge = [torch.stack([s, torch.full_like(s, k), torch.zeros_like(s)], -1) for k in (-half, half)]
  return project(cam, torch.cat(edge + [e[:, [1, 0, 2]] for e in edge]))


def composite(rgb, mask, alpha) -> torch.Tensor:
  """Floor from every pixel no robot covers, greyed so the role colours stand out, then each
  world's robots blended at alpha[e]."""
  free = (~mask).float()[..., None]
  bg = torch.where(free.sum(0) > 0, (rgb * free).sum(0) / free.sum(0).clamp_min(1), rgb.mean(0))
  out = 40 + 0.5 * bg.mean(-1, keepdim=True).expand_as(bg)
  for e in range(len(rgb)):
    out = torch.where(mask[e, ..., None], (1 - alpha[e]) * out + alpha[e] * rgb[e], out)
  return out


def plot_trails(xy, roles, ended, half, path) -> None:
  """xy: [T, N, 2 robots, 2] base positions (NaN once a world has finished)."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(7, 7))
  ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="#555", lw=1)
  for e in range(xy.shape[1]):
    for k, sign in ((0, 1.0), (1, -1.0)):
      pursuer = sign * roles[e] > 0
      color = "#d62728" if pursuer else "#4a9fe0"
      path_e = xy[:, e, k][~xy[:, e, k, 0].isnan()]
      ax.plot(*path_e.T, color=color, lw=1, alpha=0.7)
      ax.plot(*path_e[0], "o", color=color, ms=4)
      ax.plot(*path_e[-1], "x" if ended[e] == "tagged" else "s", color=color, ms=5)
  ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title=f"Base trails, {xy.shape[1]} worlds (red pursuer, blue evader; o start, x tag)")
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  print(f"saved {path}")


def main(cfg: DictConfig) -> None:
  cfg.device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  g = game(cfg)
  env = make_env(cfg, cfg.num_envs, g, eval=True, video=True)
  N, cam = env.num_envs, cfg.video.camera
  policy = load_policy(cfg.policy, cfg.device)
  std = torch.load(cfg.policy, map_location=cfg.device, weights_only=False)["actor_state_dict"]["distribution.std_param"]
  act = lambda o: (a := policy(o)) + std * torch.randn_like(a) if cfg.stochastic else policy(o)
  mj, ends_by = env.sim.mj_model, env.termination_manager.active_terms
  robot_geom = torch.tensor([mj.body(b).name.startswith(("robot/", "opponent/")) for b in mj.geom_bodyid], device=env.device)
  border = arena_outline(cam, g.rules.arena_half_extent).to(env.device)

  obs, _ = env.reset()
  steps = cfg.steps or (env.max_episode_length + 1 if cfg.freeze else env.max_episode_length)
  finished = frozen = torch.zeros(N, dtype=torch.bool, device=env.device)  # First episode over.
  rgb = mask = None
  roles, ends, wins, lengths, frames, xy = role(env).clone(), [None] * N, [None] * N, [None] * N, [], []
  with torch.no_grad():
    for t in range(steps):
      if cfg.freeze and frozen.all():
        break
      r = role(env).clone()
      pos = torch.stack([env.scene[k].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2] for k in ("robot", "opponent")], 1)
      xy.append(torch.where(finished[:, None, None], math.nan, pos).cpu())
      obs, _, terminated, truncated, _ = env.step(torch.cat([act(obs["actor"]), act(obs["opponent"])], -1))
      done = (terminated | truncated) & ~finished
      out = outcome(env, r)
      for e in done.nonzero()[:, 0].tolist():
        roles[e], lengths[e] = r[e], (t + 1) * env.step_dt
        ends[e] = "+".join(k for k in ends_by if env.termination_manager.get_term(k)[e])
        wins[e] = "pursuer" if out[e] * r[e] > 0 else "evader" if out[e] * r[e] < 0 else "draw"
      finished = finished | done
      frozen = finished if cfg.freeze else frozen
      cam_data = env.scene["world_cam"].data
      seg = cam_data.segmentation[..., 0]
      m = (cam_data.segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & robot_geom[seg.clamp_min(0)]
      live = ~frozen if rgb is not None else torch.ones_like(frozen)  # The step that ends a world shows its reset.
      rgb = cam_data.rgb.float() if rgb is None else torch.where(live[:, None, None, None], cam_data.rgb.float(), rgb)
      mask = m if mask is None else torch.where(live[:, None, None], m, mask)
      frame = composite(rgb, mask, cfg.alpha * (1 - 0.4 * frozen.float()))
      frame[border[:, 0], border[:, 1]] = 255.0
      frames.append(frame.clamp(0, 255).byte().cpu().numpy())

  summary = {"policy": cfg.policy, "stochastic": cfg.stochastic, "steps": len(frames), "worlds": [
    {"env": e, "robot_role": "pursuer" if roles[e] > 0 else "evader", "end": ends[e] or "running", "winner": wins[e], "length_s": lengths[e]}
    for e in range(N)
  ]}
  summary["wins"] = {k: sum(w["winner"] == k for w in summary["worlds"]) for k in ("pursuer", "evader", "draw")}
  summary["ends"] = {k: sum(k in (w["end"] or "").split("+") for w in summary["worlds"]) for k in ends_by + ["running"]}
  for w in summary["worlds"]:
    print(f"  env {w['env']:2d}  robot {w['robot_role']:7s}  {w['end']:22s}  winner {w['winner']}  {w['length_s'] or 0:5.2f} s")
  print("wins by role", summary["wins"], "ends", summary["ends"])

  import imageio_ffmpeg
  import mediapy

  os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
  stem = os.path.splitext(cfg.out)[0]
  with open(stem + ".json", "w") as f:
    json.dump(summary, f, indent=2)
  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  mediapy.write_video(cfg.out, frames, fps=1 / env.step_dt)
  print(f"saved {cfg.out} ({len(frames)} frames)")
  plot_trails(torch.stack(xy), roles, ends, g.rules.arena_half_extent, stem + ".png")


if __name__ == "__main__":
  run(main, "eval", GAME_SCHEMAS)
