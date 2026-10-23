"""Reward vs PPO gradient norm, and how far the weights move, for runs trained with cmd.log_norms=true.

Series per iteration, from the run's tensorboard log: Train/mean_reward (mean return of the last 100
finished episodes) and Loss/{actor,critic}_grad_norm (pre-clip norm, mean over the update's
minibatches). Pearson r (and Spearman rho) over all iterations so far, and the cumulative Pearson r:
at iteration t, over iterations 0..t. Both series trend over training, so r reflects the trends.

Weight movement, from the checkpoints: ||theta_k - theta_prev|| between consecutive model_<it>.pt over
the MLP weights and biases only, cached in <run>/plots/weight_change_mlp.json. Absolute, not relative:
the weight norm grows all run, so a relative change falls even at a constant step. The action std
(distribution.std_param) and observation normalizer statistics are excluded; the actor's mean action
std is drawn alongside, since it grows without bound in these runs.

  uv run scripts/plot_grad_reward.py RUN_DIR [--until IT] [--out OUT.png]
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

REWARD = "Train/mean_reward"
NETS = ("actor", "critic")
COLORS = {"actor": "#d95f02", "critic": "#1b9e77"}


def load(ea, until: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """tag -> (iterations, values) from an EventAccumulator (call ea.Reload() first)."""
  out = {}
  for tag in [REWARD] + [f"Loss/{k}_grad_norm" for k in NETS]:
    if tag in ea.Tags()["scalars"]:
      ev = [e for e in ea.Scalars(tag) if until is None or e.step <= until]
      out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
  return out


def weight_changes(run: str, until: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """net -> (checkpoint iteration, MLP weight change since the previous checkpoint), plus "std" ->
  (checkpoint iteration, the actor's mean action std there)."""
  import torch

  path = os.path.join(run, "plots", "weight_change_mlp.json")
  cache = json.load(open(path)) if os.path.exists(path) else {}
  its = sorted(int(re.search(r"model_(\d+)\.pt$", f).group(1)) for f in glob.glob(os.path.join(run, "model_*.pt")))
  its = [i for i in its if until is None or i <= until]
  flat = lambda sd: torch.cat([v.flatten().double() for k, v in sd.items() if k.startswith("mlp.")])
  prev = None
  for a, b in zip(its, its[1:]):
    if f"{a}-{b}" in cache:
      prev = None
      continue
    pa = prev if prev is not None else torch.load(os.path.join(run, f"model_{a}.pt"), map_location="cpu", weights_only=False)
    pb = torch.load(os.path.join(run, f"model_{b}.pt"), map_location="cpu", weights_only=False)
    cache[f"{a}-{b}"] = {k: (flat(pb[f"{k}_state_dict"]) - flat(pa[f"{k}_state_dict"])).norm().item() for k in NETS}
    cache[f"{a}-{b}"]["std"] = pb["actor_state_dict"]["distribution.std_param"].double().mean().item()
    prev = pb
  os.makedirs(os.path.dirname(path), exist_ok=True)
  json.dump(cache, open(path, "w"))
  pairs = [(b, cache[f"{a}-{b}"]) for a, b in zip(its, its[1:])]
  return {k: (np.array([b for b, _ in pairs]), np.array([c[k] for _, c in pairs])) for k in (*NETS, "std")}


def _aligned(s, a: str, b: str):
  common, ia, ib = np.intersect1d(s[a][0], s[b][0], return_indices=True)
  return common, s[a][1][ia], s[b][1][ib]


def _pearson(x, y) -> float:
  return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")


def _spearman(x, y) -> float:
  return _pearson(np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float))


def correlations(s) -> dict[str, float]:
  out = {}
  for k in NETS:
    g = f"Loss/{k}_grad_norm"
    if g not in s or REWARD not in s:
      continue
    it, r, n = _aligned(s, REWARD, g)
    out[f"{k}/pearson"], out[f"{k}/spearman"] = _pearson(r, n), _spearman(r, n)
  return out


def cumulative_r(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Pearson r over x[:t + 1], y[:t + 1] for every t (nan until it is defined)."""
  x, y = x.astype(np.float64), y.astype(np.float64)
  n = np.arange(1, len(x) + 1)
  sx, sy, sxx, syy, sxy = (np.cumsum(v) for v in (x, y, x * x, y * y, x * y))
  with np.errstate(invalid="ignore", divide="ignore"):
    return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx**2) * (n * syy - sy**2))


def plot(s, out: str, title: str, pdf: bool = True, run: str | None = None,
         until: int | None = None) -> tuple[str, dict[str, float]]:
  corr = correlations(s)
  fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))

  a = ax[0, 0]  # Reward and grad norms over training.
  a.plot(*s[REWARD], color="k", lw=1, label="mean reward")
  a.set_xlabel("iteration")
  a.set_ylabel("Train/mean_reward")
  b = a.twinx()
  for k in NETS:
    if f"Loss/{k}_grad_norm" in s:
      b.plot(*s[f"Loss/{k}_grad_norm"], color=COLORS[k], lw=0.5, alpha=0.6, label=f"{k} grad norm")
  b.set_ylabel("grad norm (pre-clip)")
  b.set_yscale("log")
  a.legend(loc="upper left", frameon=False)
  b.legend(loc="lower right", frameon=False)
  a.set_title("Reward and gradient norm over training", fontweight="bold", loc="left")

  a = ax[0, 1]  # Cumulative r(reward, grad norm).
  for k in NETS:
    g = f"Loss/{k}_grad_norm"
    if g in s:
      it, r, n = _aligned(s, REWARD, g)
      c = cumulative_r(r, n)
      a.plot(it[10:], c[10:], color=COLORS[k], lw=1.2, label=f"{k} (r = {c[-1]:.2f} at {it[-1]})")
  a.axhline(0, color="0.6", lw=0.8)
  a.set_ylim(-1, 1)
  a.set_xlabel("iteration t")
  a.set_ylabel("Pearson r(reward, grad norm) over iterations 0..t")
  a.set_title("Cumulative correlation of reward and grad norm", fontweight="bold", loc="left")
  a.legend(loc="lower left", frameon=False)

  a = ax[1, 0]  # Scatter.
  lines = []
  for k in NETS:
    g = f"Loss/{k}_grad_norm"
    if g not in s:
      continue
    it, r, n = _aligned(s, REWARD, g)
    a.scatter(n, r, s=3, color=COLORS[k], alpha=0.35, label=k)
    lines.append(f"{k}: Pearson r = {corr[f'{k}/pearson']:.2f}, Spearman ρ = {corr[f'{k}/spearman']:.2f}")
  a.set_xscale("log")
  a.xaxis.set_minor_formatter(NullFormatter())
  a.set_xlabel("grad norm (pre-clip)")
  a.set_ylabel("mean reward")
  a.legend(loc="lower right", frameon=False, markerscale=4)
  a.set_title("Reward vs grad norm, every iteration\n" + "\n".join(lines), fontweight="bold", loc="left", fontsize=10)

  a = ax[1, 1]  # MLP weight movement between checkpoints, with the action std.
  if run:
    wc = weight_changes(run, until)
    for k in NETS:
      if len(wc[k][0]):
        a.plot(*wc[k], "o-", ms=3, color=COLORS[k], label=f"{k} MLP ‖Δθ‖")
    a.set_ylim(bottom=0)
    b = a.twinx()
    if len(wc["std"][0]):
      b.plot(*wc["std"], "--", color="#7570b3", lw=1.5, label="actor mean action std")
    b.set_ylabel("actor mean action std (dashed)")
    b.set_ylim(bottom=0)
    b.legend(loc="upper right", frameon=False)
  a.set_xlabel("checkpoint (iteration)")
  a.set_ylabel("‖θ_k − θ_prev‖, MLP weights, per checkpoint interval")
  a.set_title("How far the network weights move between checkpoints", fontweight="bold", loc="left")
  a.legend(loc="upper left", frameon=False)

  fig.suptitle(f"{title}: reward, PPO gradient norm and weight movement (through iteration {int(s[REWARD][0][-1])})",
               x=0.01, ha="left")
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  fig.savefig(out, dpi=130)
  if pdf:
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
  plt.close(fig)
  print(f"saved {out}")
  return out, corr


def main() -> None:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run")
  ap.add_argument("--until", type=int, default=None)
  ap.add_argument("--out", default=None)
  a = ap.parse_args()
  run = os.path.abspath(a.run)
  s = load(EventAccumulator(run, size_guidance={"scalars": 0}).Reload(), a.until)
  name = f"{os.path.basename(os.path.dirname(run))} / {os.path.basename(run)}"
  _, corr = plot(s, a.out or f"logs/plots/grad_reward_{os.path.basename(run)}.png", name, run=run, until=a.until)
  print(json.dumps(corr, indent=2))


if __name__ == "__main__":
  main()
