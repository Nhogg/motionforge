"""Rolling evaluator: score (and render) every new training checkpoint, newest first.

  python -m mofo.watch --root logs/mofo/tag/v3 --jobs 4701,4702 --opponent STAGE1.pt [--shard 0/2] [--wandb online]

Each pass evaluates checkpoints under ``<root>/<run>/model_<it>.pt`` that lack
``<run>/eval/model_<it>.json`` with ``scripts/mofo_train.py eval`` (256-env stats; a 16-env
MP4 for runs whose seed is in ``--video-seeds``), then redraws ``<root>/progress.png``. It
exits once none of ``--jobs`` is queued and nothing is left to evaluate.

With ``--wandb online|offline`` each owned run also streams to wandb (``WandbRun``): its
tensorboard scalars, rolling evals (``eval/*``), and the chained final evals (``final/*``,
``final_legacy/*``), all against ``iteration``.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import time

EVAL = [".venv/bin/python", "scripts/mofo_train.py", "cmd=eval"]
RENAME = {"win_rate": "win", "win_rate_pursuer": "pursuer_win", "win_rate_evader": "evader_win"}
# tag-rules' ScriptedPolicy tie fix went live 2026-10-15 20:09:04 CDT: eval JSONs written before
# it faced the old scripted opponent (scripted_version 0). Those are re-run (``refix``) and the
# old ones kept as ``model_<it>_prefix.json``. +120 s: an eval (<= ~90 s) that started before the
# fix imported the old policy but may have finished after it.
SCRIPTED_FIX = 1792112944 + 120


def scripted_version(path: str) -> int:
  return int(os.path.getmtime(path) >= SCRIPTED_FIX)


def prefix_path(path: str) -> str:
  return path[: -len(".json")] + "_prefix.json"


def eval_row(d: dict, prefix: str) -> dict:
  """Scalar eval stats (and the min-distance and naturalness summaries) under ``prefix/``."""
  row = {}
  for k, v in d.items():
    k = RENAME.get(k, k)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      row[f"{prefix}/{k}"] = v
    elif (k.startswith("min_distance") or k == "naturalness") and isinstance(v, dict):
      row.update({f"{prefix}/{k}/{a}": b for a, b in v.items() if isinstance(b, (int, float))})
  return row


class WandbRun:
  """One wandb run per training run dir, id ``<group>-<name>``, on an ``iteration`` axis
  (evals arrive newest-first, so no wandb step). Logs the run's tensorboard scalars
  incrementally, ``<run>/eval/model_<it>.{json,mp4}`` as ``eval/*``, and the chained
  ``<root>/eval_<name>_<it>[_legacy].{json,mp4}`` as ``final/*`` / ``final_legacy/*``.
  What was sent is kept in ``<run>/wandb_<group>.json``, so restarts resume without duplicates.
  ``shared=True`` joins a run the trainer owns (wandb shared mode) instead of creating it.
  """

  def __init__(self, root: str, run: str, group: str, mode: str = "online", shared: bool = False):
    import wandb
    import yaml

    self.wandb, self.root, self.run, self.name, self.shared = wandb, root, run, os.path.basename(run), shared
    cfg = yaml.safe_load(open(os.path.join(run, ".hydra", "config.yaml")))
    lg, arm = cfg.get("league", {}), self.name.rsplit("_s", 1)[0]
    tags = [arm, f"shaping={'on' if cfg['shaping']['enabled'] else 'off'}", f"selfplay_frac={lg.get('selfplay_frac')}",
            f"sampling={lg.get('sampling')}", f"spawn={'spread' if cfg['reset']['placement'] == 'uniform' else 'legacy'}"]
    # Shared: the trainer owns the run (its tags and config); join as a secondary writer only.
    own = {} if shared else {"tags": tags, "config": cfg}
    settings = wandb.Settings(mode="shared", x_primary=False, x_label="eval") if shared else None
    self.wb = wandb.init(project="mofo", id=f"{group}-{self.name}", name=self.name, group=group, resume="allow",
                         mode=mode, reinit="create_new", dir=run, settings=settings, **own)
    self.wb.define_metric("iteration")
    self.wb.define_metric("*", step_metric="iteration")
    self.state_path = os.path.join(run, f"wandb_{group}.json")
    self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {"tb_step": -1, "sent": []}
    self.sent = set(self.state["sent"])

  def _files(self):
    """(key, path, prefix, iteration): rolling then final evals, json and mp4."""
    for f in glob.glob(os.path.join(self.run, "eval", "model_*")):
      if m := re.search(r"model_(\d+)\.(json|mp4)$", f):
        yield f, f, "eval", int(m.group(1))
    for f in glob.glob(os.path.join(self.root, f"eval_{self.name}_*")):
      if m := re.search(rf"eval_{re.escape(self.name)}_(\d+)(_legacy)?\.(json|mp4)$", f):
        yield f, f, "final_legacy" if m.group(2) else "final", int(m.group(1))

  def sync(self) -> int:
    """Send everything new; returns the number of rows logged."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows = {}
    ea = None if self.shared else EventAccumulator(self.run, size_guidance={"scalars": 0}).Reload()  # Shared: the trainer logs these.
    for tag in ea.Tags()["scalars"] if ea else ():
      for e in ea.Scalars(tag):
        if e.step > self.state["tb_step"]:
          rows.setdefault(e.step, {})[tag] = e.value
    for it in sorted(rows)[:-1]:  # The newest step may still be filling in; take it next pass.
      self.wb.log({**rows[it], "iteration": it})
      self.state["tb_step"] = it
    n = max(len(rows) - 1, 0)
    for key, path, prefix, it in sorted(self._files(), key=lambda x: x[3]):
      if key in self.sent and path.endswith(".json") and os.path.exists(prefix_path(path)):
        key, prefix = f"{key}:fixed", f"{prefix}_fixed"  # Re-run of a pre-fix eval: its own series.
      if key in self.sent or time.time() - os.path.getmtime(path) < 30:
        continue
      if path.endswith(".json"):
        d = json.load(open(path))
        row = {} if "error" in d else {**eval_row(d, prefix), f"{prefix}/scripted_version": scripted_version(path)}
      else:
        row = {f"{prefix}/video": self.wandb.Video(path, format="mp4")}
      if row:
        self.wb.log({**row, "iteration": it})
        n += 1
      self.sent.add(key)
    self.state["sent"] = sorted(self.sent)
    json.dump(self.state, open(self.state_path, "w"))
    return n

  def finish(self) -> None:
    self.wb.finish()


def owned(root: str, shard: tuple[int, int]) -> list[str]:
  """Runs ``<arm>_s<seed>`` with seed % n == k: ownership stays fixed as new run dirs appear."""
  runs = sorted(d for d in glob.glob(os.path.join(root, "*_s*")) if os.path.isdir(d))
  return [r for r in runs if int(r.rsplit("_s", 1)[1]) % shard[1] == shard[0]]


def pending(root: str, shard: tuple[int, int]) -> list[tuple[int, str, str, bool]]:
  """(iteration, run dir, ckpt, refix): checkpoints without an eval, newest first and settled
  files only; then pre-fix evals to re-run against the fixed scripted opponent."""
  new, refix = [], []
  for run in owned(root, shard):
    for ckpt in glob.glob(os.path.join(run, "model_*.pt")):
      it = int(re.search(r"model_(\d+)\.pt$", ckpt).group(1))
      out = os.path.join(run, "eval", f"model_{it}.json")
      if not os.path.exists(out):
        if time.time() - os.path.getmtime(ckpt) > 30:
          new.append((it, run, ckpt, False))
      elif not scripted_version(out) and not os.path.exists(prefix_path(out)) and "error" not in json.load(open(out)):
        refix.append((it, run, ckpt, True))
  return sorted(new, key=lambda x: -x[0]) + sorted(refix, key=lambda x: -x[0])


def evaluate(it: int, run: str, ckpt: str, args, refix: bool = False) -> float:
  """Stats (+ video for --video-seeds) of one checkpoint. ``refix``: stats only, then the
  pre-fix JSON moves to ``*_prefix.json`` and the new one takes its place."""
  out = os.path.join(run, "eval", f"model_{it}")
  os.makedirs(os.path.dirname(out), exist_ok=True)
  common = [f"opponent={args.opponent}", f"cmd.policy={ckpt}", *args.override]
  t = time.time()
  if refix:
    r = subprocess.run(EVAL + ["num_envs=256", f"cmd.out={out}_refix.json", *common], capture_output=True, text=True)
    if r.returncode:
      print(f"[watch] FAILED refix {out}\n{r.stderr[-2000:]}", flush=True)
      json.dump({"error": r.stderr[-2000:]}, open(f"{out}_prefix.json", "w"))  # Marks it done; keeps the old eval.
    else:
      os.rename(f"{out}.json", f"{out}_prefix.json")
      os.rename(f"{out}_refix.json", f"{out}.json")
    return time.time() - t
  cmds = [EVAL + ["num_envs=256", f"cmd.out={out}.json", *common]]
  if int(run.rsplit("_s", 1)[1]) in args.video_seeds:
    cmds.append(EVAL + ["num_envs=16", f"cmd.out={out}_video.json", f"cmd.video={out}.mp4", *common])
  for cmd in cmds:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
      print(f"[watch] FAILED {' '.join(cmd)}\n{r.stderr[-2000:]}", flush=True)
      json.dump({"error": r.stderr[-2000:]}, open(f"{out}.json", "w"))  # Don't retry forever.
      break
  return time.time() - t


def scripted_share(run: str) -> tuple[list[int], list[float]]:
  """Actual env share of the scripted opponent per iteration, from the run's tensorboard log."""
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  ea = EventAccumulator(run, size_guidance={"scalars": 0}).Reload()
  if "league/envs/scripted" not in ea.Tags()["scalars"]:
    return [], []
  ev = ea.Scalars("league/envs/scripted")
  return [e.step for e in ev], [e.value for e in ev]


def plot(root: str) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  keys = [("win_rate_pursuer", "pursuer win rate"), ("win_rate_evader", "evader win rate"),
          ("p95_planar_speed", "p95 planar speed (m/s)"), ("fall_rate", "fall rate")]
  natural = [("pose_dev", "pose deviation"), ("torso_tilt", "torso tilt"), ("foot_slip_speed", "foot slip speed (m/s)"),
             ("base_height", "base height (m)"), ("feet_air_time", "feet air time (s)")]  # Mean of both roles.
  fig, grid = plt.subplots(2, 5, figsize=(24, 9))
  axes, nat_axes = grid
  for run in sorted(d for d in glob.glob(os.path.join(root, "*_s*")) if os.path.isdir(d)):
    rows = []
    for f in glob.glob(os.path.join(run, "eval", "model_*.json")):
      m = re.search(r"model_(\d+)\.json$", f)  # Skips _video, _prefix and in-flight _refix files.
      d = json.load(open(f)) if m else {"error": None}
      if "error" not in d:
        rows.append((int(m.group(1)), d))
    rows.sort(key=lambda x: x[0])
    name = os.path.basename(run)
    for ax, (k, _) in zip(axes, keys):
      if rows:
        ax.plot([i for i, _ in rows], [d[k] for _, d in rows], marker="o", markersize=3, linewidth=1.2, label=name)
    for ax, (k, _) in zip(nat_axes, natural):
      pts = [(i, [v for r, v in d["naturalness"].items() if r.rsplit("_", 1)[0] == k]) for i, d in rows if d.get("naturalness")]
      if pts := [(i, sum(v) / len(v)) for i, v in pts if v]:
        ax.plot(*zip(*pts), marker="o", markersize=3, linewidth=1.2, label=name)
    its, share = scripted_share(run)
    if its:
      axes[4].plot(its, share, linewidth=1, label=name)
  for ax, (_, title) in zip(list(axes) + list(nat_axes), keys + [(None, "scripted opponent env share (training)")] + natural):
    ax.set(title=title, xlabel="iteration")
    ax.spines[["top", "right"]].set_visible(False)
  axes[0].legend(fontsize=7, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(root, "progress.png"), dpi=110)
  plt.close(fig)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--root", required=True)
  p.add_argument("--opponent", required=True, help="stage-1 ckpt for the scripted opponent")
  p.add_argument("--jobs", default="", help="training job ids; exit when none is queued")
  p.add_argument("--shard", default="0/1", help="k/n: own the runs whose seed %% n == k")
  p.add_argument("--video-seeds", type=lambda s: [int(x) for x in s.split(",") if x], default=[1, 2])
  p.add_argument("--interval", type=float, default=60, help="seconds between idle passes")
  p.add_argument("--wandb", choices=("online", "offline", "disabled"), default="online")
  p.add_argument("--wandb-group", default=None, help="wandb group and run-id prefix (default: basename of --root)")
  p.add_argument("--wandb-shared", action="store_true", help="join the trainer's run (it logs training; we log evals)")
  p.add_argument("override", nargs="*", help="extra eval overrides, e.g. reset=legacy")
  args = p.parse_args()
  shard = tuple(int(x) for x in args.shard.split("/"))
  jobs = [j for j in args.jobs.split(",") if j]
  group, wb = args.wandb_group or os.path.basename(os.path.normpath(args.root)), {}

  def sync():  # Attach to each owned run once its config exists, then send what's new.
    if args.wandb == "disabled":
      return
    for run in owned(args.root, shard):
      if run not in wb and os.path.exists(os.path.join(run, ".hydra", "config.yaml")):
        wb[run] = WandbRun(args.root, run, group, args.wandb, shared=args.wandb_shared)
        print(f"[watch] wandb {wb[run].wb.url or wb[run].wb.id}", flush=True)
      if run in wb:
        wb[run].sync()

  while True:
    sync()
    todo = pending(args.root, shard)
    if todo:
      it, run, ckpt, refix = todo[0]
      dt = evaluate(it, run, ckpt, args, refix)
      print(f"[watch] {os.path.basename(run)} model_{it}{' refix' if refix else ''}: {dt:.0f}s; {len(todo) - 1} pending", flush=True)
      plot(args.root)
      continue
    queued = jobs and subprocess.run(["squeue", "-h", "-j", ",".join(jobs)], capture_output=True, text=True).stdout.strip()
    if not queued:
      plot(args.root)
      time.sleep(31)  # Let the last files settle, then a final sync.
      sync()
      for w in wb.values():
        w.finish()
      return
    time.sleep(args.interval)


if __name__ == "__main__":
  main()
