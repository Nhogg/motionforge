"""Log per-checkpoint figures and a video for one or more runs to wandb, to scrub through training.

  uv run scripts/wandb_plots.py RUN_DIR [RUN_DIR ...] [--follow] [--num-envs 4096] [--wandb online]

Per run dir (tag or task=goal) a companion wandb run ``<group>-<name>-viz`` in the training run's
group. Each checkpoint ``model_<it>.pt`` is logged at step ``it`` (deterministic actions, fixed
reset seed, so every checkpoint sees the same spawns):

  plots/states, plots/velocity_heading, plots/torque_corr   plot_states (one episode in num_envs worlds)
  plots/elo                                                 tag: plot_elo, this checkpoint vs every earlier one
  plots/role_matrix                                         tag: the same games by role (pursuer tag rate below
                                                            the diagonal, evader not-tagged rate above)
  plots/league_matrix                                       tag: the training league's EMA win rate + PFSP,
                                                            from the run's tensorboard log, up to ``it``
  plots/grad_reward, corr/*                                 cmd.log_norms runs: plot_grad_reward up to ``it``
  video                                                     8 worlds overlaid as in scripts/eval.py (goal
                                                            task: each world's current goal in yellow). Tag:
                                                            reset seed --seed + it (new spawns and roles per
                                                            checkpoint); worlds 0-1 self-play, worlds 2-7 vs
                                                            earlier checkpoints of the run drawn at random
                                                            (listed in the caption and <stem>_video.json)
  stats/*, elo/*                                            summary scalars

Figures, npz and mp4 go to ``<run>/plots/``; logged iterations are kept in
``<run>/plots/wandb_viz.json``, so a restart skips them. Checkpoints are taken oldest first (wandb
steps must increase). ``--follow`` polls until each run has logged its final checkpoint
(``model_<cmd.iterations - 1>.pt``); a run dir without ``.hydra/`` yet (a queued job) is picked up once it appears.

A CUDA device-side assert (seen once, intermittently, in a reused env) kills the process, so run it in
a restart loop, e.g. ``bash -c 'for i in $(seq 20); do python scripts/wandb_plots.py ... && break; done'``: attempts are counted in ``<run>/plots/wandb_viz_attempts.json``
and a checkpoint that has already been attempted ``MAX_ATTEMPTS`` times is skipped (and printed).
"""

import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_states  # noqa: E402  (sets MUJOCO_GL before mjlab loads)

SETTLE_S = 30  # A checkpoint younger than this may still be being written.
MAX_ATTEMPTS = 2  # Per checkpoint, across restarts.
VIDEO_WORLDS, GOAL_VIDEO_STEPS = 8, 500  # Goal episodes run to 20 s: show the first 10.
CAMERA = {"back": 7.0, "up": 8.0, "width": 640, "height": 480, "data_types": ["rgb", "segmentation"]}


def video_opponents(run: str, it: int, n: int = VIDEO_WORLDS, n_self: int = 2) -> list[str | None]:
  """Per video world: None (self-play) for the first n_self, else an earlier checkpoint of the run
  drawn uniformly (seeded by it, so a re-log gives the same draw)."""
  import random

  earlier = sorted((int(m.group(1)), f) for f in glob.glob(os.path.join(run, "model_*.pt"))
                   if (m := re.search(r"model_(\d+)\.pt$", f)) and int(m.group(1)) < it)
  rng = random.Random(it)
  return [None] * n_self + [rng.choice(earlier)[1] if earlier else None for _ in range(n - n_self)]


def render(env, ckpt: str, out: str, cam, half: float, seed: int = 0, alpha: float = 0.5,
           opponents: list[str | None] | None = None) -> str:
  """scripts/eval.py's overlay: every world's robots blended over a shared floor, each world held
  on its last frame once its first episode ends. ``opponents`` (tag): per world, a checkpoint that
  drives the opponent seat (None: this checkpoint, i.e. self-play)."""
  import imageio_ffmpeg
  import mediapy
  import mujoco
  import torch
  from eval import arena_outline, composite, project

  from mofo.opponent import load_policy

  dev, n = env.device, env.num_envs
  policy = load_policy(ckpt, dev)
  solo = "opponent" not in env.scene.entities
  opp = {}  # Opponent checkpoint -> (policy, its worlds).
  for w, o in enumerate(opponents or []):
    if o is not None and o != ckpt:
      opp.setdefault(o, [None, []])[1].append(w)
  for o in opp:
    opp[o][0] = load_policy(o, dev)
  mj = env.sim.mj_model
  robot_geom = torch.tensor([mj.body(b).name.startswith(("robot/", "opponent/")) for b in mj.geom_bodyid], device=dev)
  border = arena_outline(cam, half).to(dev)
  steps = GOAL_VIDEO_STEPS if solo else env.max_episode_length + 1
  obs, _ = env.reset(seed=seed)
  finished = torch.zeros(n, dtype=torch.bool, device=dev)
  rgb = mask = None
  frames = []
  with torch.no_grad():
    for _ in range(steps):
      if finished.all():
        break
      if solo:
        a = policy(obs["actor"])
      else:
        b = policy(obs["opponent"])
        for p, ws in opp.values():
          b[ws] = p(obs["opponent"][ws])
        a = torch.cat([policy(obs["actor"]), b], -1)
      obs, _, term, trunc, _ = env.step(a)
      first = rgb is None
      finished = finished | term | trunc
      d = env.scene["world_cam"].data
      seg = d.segmentation[..., 0]
      m = (d.segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & robot_geom[seg.clamp_min(0)]
      live = torch.ones_like(finished) if first else ~finished  # The step that ends a world shows its reset.
      rgb = d.rgb.float() if first else torch.where(live[:, None, None, None], d.rgb.float(), rgb)
      mask = m if first else torch.where(live[:, None, None], m, mask)
      frame = composite(rgb, mask, alpha * (1 - 0.4 * finished.float()))
      frame[border[:, 0], border[:, 1]] = 255.0
      if solo:  # Current goals of the worlds still playing.
        from mofo.tasks.goal import goal_term

        g = goal_term(env).goal[~finished] - env.scene.env_origins[~finished, :2]
        c = project(cam, torch.cat([g, torch.zeros_like(g[:, :1])], -1)).to(dev)
        for dr in range(-2, 3):
          for dc in range(-2, 3):
            rc = c + torch.tensor([dr, dc], device=dev)
            rc = rc[((rc >= 0) & (rc < torch.tensor([cam.height, cam.width], device=dev))).all(-1)]
            frame[rc[:, 0], rc[:, 1]] = torch.tensor([255.0, 210.0, 0.0], device=dev)
      frames.append(frame.clamp(0, 255).byte().cpu().numpy())
  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  mediapy.write_video(out, frames, fps=1 / env.step_dt)
  print(f"saved {out} ({len(frames)} frames)")
  return out


class VizRun:
  def __init__(self, run: str, args):
    import wandb
    import yaml

    self.run, self.args = os.path.abspath(run), args
    self.cfg = yaml.safe_load(open(os.path.join(self.run, ".hydra", "config.yaml")))
    self.tag = self.cfg.get("task", "tag") == "tag"
    self.final = self.cfg["cmd"]["iterations"] - 1
    self.name, group = os.path.basename(self.run), os.path.basename(os.path.dirname(self.run))
    self.title = f"{group} / {self.name}"
    self.out = os.path.join(self.run, "plots")
    os.makedirs(self.out, exist_ok=True)
    self.state_path = os.path.join(self.out, "wandb_viz.json")
    self.done = json.load(open(self.state_path)) if os.path.exists(self.state_path) else []
    self.attempts_path = os.path.join(self.out, "wandb_viz_attempts.json")
    self.attempts = json.load(open(self.attempts_path)) if os.path.exists(self.attempts_path) else {}
    self.wb = wandb.init(
      project=self.cfg["wandb"]["project"], entity=self.cfg["wandb"].get("entity"), id=f"{group}-{self.name}-viz",
      name=f"{self.name}-viz", group=group, job_type="viz", resume="allow", mode=args.wandb, dir=self.out,
      reinit="create_new", tags=["viz", *self.cfg["wandb"].get("tags", [])],
      config={"train_run": f"{group}-{self.name}", "num_envs": args.num_envs, "reset_seed": args.seed,
              "actions": "deterministic", "video_worlds": VIDEO_WORLDS, **self.cfg},
    )
    self.wandb, self.env, self.video_env, self.ea = wandb, None, None, None
    if self.tag:
      import plot_elo

      self.elo = plot_elo.Tournament(self.run)

  def pending(self) -> list[tuple[int, str]]:
    last = max(self.done, default=-1)
    out = []
    for ckpt in glob.glob(os.path.join(self.run, "model_*.pt")):
      it = int(re.search(r"model_(\d+)\.pt$", ckpt).group(1))
      if it > last and time.time() - os.path.getmtime(ckpt) > SETTLE_S:
        out.append((it, ckpt))
    return sorted(out)

  def events(self):
    """The run's tensorboard log, reloaded (incrementally)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if self.ea is None:
      self.ea = EventAccumulator(self.run, size_guidance={"scalars": 0})
    return self.ea.Reload()

  def league_matrix(self, it: int, stem: str) -> str | None:
    import plot_league_matrix as plm

    hist = plm.tb_history(self.events())
    every = self.cfg["league"].get("snapshot_every") or 250
    if not any(k <= it for k in hist) or it < every:
      return None
    return plm.plot(hist, plm.league_cfg(self.cfg["league"]), stem + "_league.png", self.title, every, it, pdf=False)

  def log(self, it: int, ckpt: str) -> None:
    import numpy as np
    import plot_elo
    from omegaconf import OmegaConf

    t0, a = time.time(), self.args
    n = self.attempts.get(str(it), 0)
    if n >= MAX_ATTEMPTS:
      print(f"[{self.name}] skipping iteration {it}: crashed {n} times", flush=True)
      self.done.append(it)
      json.dump(self.done, open(self.state_path, "w"))
      return
    self.attempts[str(it)] = n + 1
    json.dump(self.attempts, open(self.attempts_path, "w"))
    if self.env is None:  # Built on first use; one per run (its own game config).
      self.env = plot_states.load_env(self.run, a.num_envs, a.device)
    stem = os.path.join(self.out, f"model_{it}")
    if os.path.exists(stem + ".npz"):  # Same rollout (fixed seed, deterministic): reuse it.
      h = dict(np.load(stem + ".npz", allow_pickle=True))
    else:
      h = plot_states.rollout(self.env, ckpt, seed=a.seed)
      np.savez(stem + ".npz", **h)
    paths = plot_states.plot_all(h, stem + ".png", pdf=False)
    row = {f"stats/{k}": v for k, v in plot_states.summary(h).items()}
    if self.tag:
      row.update(self.elo.add(self.env, ckpt, a.seed))
      paths["elo"] = plot_elo.plot(self.elo.state, stem + "_elo.png", self.title, pdf=False)
      paths["role_matrix"] = plot_elo.plot_roles(self.elo.state, stem + "_roles.png", self.title, pdf=False)
      if p := self.league_matrix(it, stem):
        paths["league_matrix"] = p
    if self.cfg["cmd"].get("log_norms") and it >= 100:
      import plot_grad_reward as pgr

      series = pgr.load(self.events(), it)
      if pgr.REWARD in series:
        paths["grad_reward"], corr = pgr.plot(series, stem + "_grad_reward.png", self.title, pdf=False, run=self.run, until=it)
        row.update({f"corr/{k}": v for k, v in corr.items()})
    row.update({f"plots/{k}": self.wandb.Image(p) for k, p in paths.items()})
    if self.video_env is None:
      self.video_env = plot_states.load_env(self.run, VIDEO_WORLDS, a.device, camera=CAMERA)
      self.cam = OmegaConf.merge(OmegaConf.load(os.path.join(self.run, ".hydra", "config.yaml")).video.camera, CAMERA)
    if self.tag:
      opps = video_opponents(self.run, it)
      names = ["self" if o is None else os.path.basename(o)[:-3] for o in opps]
      json.dump({"reset_seed": a.seed + it, "opponents": names}, open(stem + "_video.json", "w"))
      mp4 = render(self.video_env, ckpt, stem + ".mp4", self.cam, float(h["half"]), a.seed + it, opponents=opps)
      row["video"] = self.wandb.Video(mp4, format="mp4", caption=f"seed {a.seed + it}; opponents: " + ", ".join(names))
    else:
      mp4 = render(self.video_env, ckpt, stem + ".mp4", self.cam, float(h["half"]), a.seed + it)
      row["video"] = self.wandb.Video(mp4, format="mp4", caption=f"seed {a.seed + it}")
    self.wb.log({**row, "iteration": it}, step=it)
    self.done.append(it)
    json.dump(self.done, open(self.state_path, "w"))
    print(f"[{self.name}] logged iteration {it} ({time.time() - t0:.0f} s)", flush=True)

  @property
  def finished(self) -> bool:
    return self.final in self.done


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("runs", nargs="+", help="Training run dirs (holding .hydra/ and model_*.pt).")
  ap.add_argument("--follow", action="store_true", help="Keep polling until every run logged its final checkpoint.")
  ap.add_argument("--poll", type=float, default=60.0, help="Seconds between polls with --follow.")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--seed", type=int, default=0, help="Reset seed: the same spawns for every checkpoint.")
  ap.add_argument("--device", default=None)
  ap.add_argument("--wandb", choices=("online", "offline"), default="online")
  a = ap.parse_args()
  runs, waiting = [], list(a.runs)
  while True:
    for r in [r for r in waiting if os.path.exists(os.path.join(r, ".hydra", "config.yaml"))]:  # Runs that started late.
      runs.append(VizRun(r, a))
      waiting.remove(r)
    # Round-robin over runs, one checkpoint each, so no run waits behind another's backlog.
    todo = False
    for r in runs:
      if p := r.pending():
        r.log(*p[0])
        todo = True
    if not a.follow or (not waiting and all(r.finished for r in runs)):
      if not todo:
        break
      continue
    if not todo:
      time.sleep(a.poll)
  for r in runs:
    r.wb.finish()


if __name__ == "__main__":
  main()
