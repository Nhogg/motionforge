"""Rolling evaluator: score (and render) every new training checkpoint, newest first.

  python -m mofo.watch --root logs/mofo/tag/v3 --jobs 4701,4702 --opponent STAGE1.pt [--shard 0/2]

Each pass evaluates checkpoints under ``<root>/<run>/model_<it>.pt`` that lack
``<run>/eval/model_<it>.json`` with ``scripts/mofo_train.py eval`` (256-env stats; a 16-env
MP4 for runs whose seed is in ``--video-seeds``), then redraws ``<root>/progress.png``. It
exits once none of ``--jobs`` is queued and nothing is left to evaluate.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import time

EVAL = [".venv/bin/python", "scripts/mofo_train.py", "cmd=eval"]
# tag-rules' ScriptedPolicy tie fix went live 2026-10-15 20:09:04 CDT: eval JSONs written before
# it faced the old scripted opponent (scripted_version 0). Those are re-run (``refix``) and the
# old ones kept as ``model_<it>_prefix.json``. +120 s: an eval (<= ~90 s) that started before the
# fix imported the old policy but may have finished after it.
SCRIPTED_FIX = 1792112944 + 120


def scripted_version(path: str) -> int:
  return int(os.path.getmtime(path) >= SCRIPTED_FIX)


def prefix_path(path: str) -> str:
  return path[: -len(".json")] + "_prefix.json"


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
  p.add_argument("override", nargs="*", help="extra eval overrides, e.g. reset=legacy")
  args = p.parse_args()
  shard = tuple(int(x) for x in args.shard.split("/"))
  jobs = [j for j in args.jobs.split(",") if j]
  while True:
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
      return
    time.sleep(args.interval)


if __name__ == "__main__":
  main()
