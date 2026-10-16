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


def pending(root: str, shard: tuple[int, int]) -> list[tuple[int, str, str]]:
  """(iteration, run dir, ckpt) without an eval, newest iteration first; settled files only."""
  runs = sorted(d for d in glob.glob(os.path.join(root, "*_s*")) if os.path.isdir(d))
  out = []
  for i, run in enumerate(runs):
    if i % shard[1] != shard[0]:
      continue
    for ckpt in glob.glob(os.path.join(run, "model_*.pt")):
      it = int(re.search(r"model_(\d+)\.pt$", ckpt).group(1))
      if not os.path.exists(os.path.join(run, "eval", f"model_{it}.json")) and time.time() - os.path.getmtime(ckpt) > 30:
        out.append((it, run, ckpt))
  return sorted(out, key=lambda x: -x[0])


def evaluate(it: int, run: str, ckpt: str, args) -> float:
  out = os.path.join(run, "eval", f"model_{it}")
  os.makedirs(os.path.dirname(out), exist_ok=True)
  common = [f"opponent={args.opponent}", f"cmd.policy={ckpt}", *args.override]
  t = time.time()
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
  fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
  for run in sorted(d for d in glob.glob(os.path.join(root, "*_s*")) if os.path.isdir(d)):
    rows = []
    for f in glob.glob(os.path.join(run, "eval", "model_*.json")):
      if f.endswith("_video.json"):
        continue
      d = json.load(open(f))
      if "error" not in d:
        rows.append((int(re.search(r"model_(\d+)\.json$", f).group(1)), d))
    rows.sort(key=lambda x: x[0])
    name = os.path.basename(run)
    for ax, (k, _) in zip(axes, keys):
      if rows:
        ax.plot([i for i, _ in rows], [d[k] for _, d in rows], marker="o", markersize=3, linewidth=1.2, label=name)
    its, share = scripted_share(run)
    if its:
      axes[4].plot(its, share, linewidth=1, label=name)
  for ax, (_, title) in zip(axes, keys + [(None, "scripted opponent env share (training)")]):
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
  p.add_argument("--shard", default="0/1", help="k/n: evaluate runs with index %% n == k")
  p.add_argument("--video-seeds", type=lambda s: [int(x) for x in s.split(",") if x], default=[1, 2])
  p.add_argument("--interval", type=float, default=60, help="seconds between idle passes")
  p.add_argument("override", nargs="*", help="extra eval overrides, e.g. reset=legacy")
  args = p.parse_args()
  shard = tuple(int(x) for x in args.shard.split("/"))
  jobs = [j for j in args.jobs.split(",") if j]
  while True:
    todo = pending(args.root, shard)
    if todo:
      it, run, ckpt = todo[0]
      dt = evaluate(it, run, ckpt, args)
      print(f"[watch] {os.path.basename(run)} model_{it}: {dt:.0f}s; {len(todo) - 1} pending", flush=True)
      plot(args.root)
      continue
    queued = jobs and subprocess.run(["squeue", "-h", "-j", ",".join(jobs)], capture_output=True, text=True).stdout.strip()
    if not queued:
      plot(args.root)
      return
    time.sleep(args.interval)


if __name__ == "__main__":
  main()
