"""Elo of a tag run's checkpoints from deterministic head-to-head games between them.

Each checkpoint plays every earlier one in a single batched rollout: the worlds are split evenly
among the earlier checkpoints, the new one takes the robot seat and the old one the opponent seat,
roles are drawn 50/50 per world (tag's learner_role=random), and every world plays one episode
from a fixed reset seed. A game scores 1 for the side whose pursuer tags, 0 for the other, 0.5
for anything else (timeout, fall, exit: draws under tag_scratch_lim). Results accumulate in
``<run>/plots/elo.json``; ratings are refit on all of them each time (Bradley-Terry MLE on the Elo
scale, each pairing smoothed with one drawn pseudo-game so a clean sweep stays finite), anchored at
the run's first checkpoint (model_0, the untrained policy) = 0. Games are also split by the new
checkpoint's role, for the role matrix (role_matrix): its pursuer tag rate and its evader
not-tagged rate against each earlier checkpoint.

  uv run scripts/plot_elo.py RUN_DIR [--num-envs 4096]   # every checkpoint not yet in elo.json
"""

import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_states  # noqa: E402  (sets MUJOCO_GL before mjlab loads)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ELO = 400 / math.log(10)  # Elo points per logit.


def ckpt_iter(path: str) -> int:
  return int(re.search(r"model_(\d+)\.pt$", path).group(1))


def head_to_head(env, ckpt: str, opponents: list[str], seed: int = 0) -> list[tuple[float, int, int, int, int, int]]:
  """Per opponent: (score sum, games, games as pursuer, tags made, games as evader, times
  tagged) of ``ckpt``, one episode per world."""
  from mofo.opponent import load_policy
  from mofo.tasks.tag import outcome, role

  dev, n, k = env.device, env.num_envs, len(opponents)
  me, them = load_policy(ckpt, dev), [load_policy(p, dev) for p in opponents]
  group = torch.arange(n, device=dev) % k
  idx = [(group == g).nonzero()[:, 0] for g in range(k)]
  score = torch.full((n,), 0.5, device=dev)
  obs, _ = env.reset(seed=seed)
  pursuer = role(env) > 0  # Drawn at reset; held for the whole (single) episode.
  live = torch.ones(n, dtype=torch.bool, device=dev)
  with torch.no_grad():
    while live.any():
      r = role(env).clone()
      opp = torch.empty(n, env.action_manager.get_term("opponent").action_dim, device=dev)
      for g, ids in enumerate(idx):
        opp[ids] = them[g](obs["opponent"][ids])
      obs, _, term, trunc, _ = env.step(torch.cat([me(obs["actor"]), opp], -1))
      done = (term | trunc) & live
      out = outcome(env, r)  # +1 robot seat wins, -1 loses, 0 draw / still running.
      score[done] = 0.5 + 0.5 * out[done]
      live &= ~done
  won, lost = score == 1.0, score == 0.0  # As pursuer a win is a tag; as evader a loss is being tagged.
  return [(score[ids].sum().item(), len(ids), int(pursuer[ids].sum()), int((won & pursuer)[ids].sum()),
           int((~pursuer)[ids].sum()), int((lost & ~pursuer)[ids].sum())) for ids in idx]


def fit(players: list[int], games: list[list]) -> np.ndarray:
  """Elo per player (anchored players[0] = 0) from [i, j, score_i, n] rows, by Newton on the
  Bradley-Terry log-likelihood, each pairing with one extra drawn game."""
  pos = {p: a for a, p in enumerate(players)}
  agg: dict[tuple[int, int], list[float]] = {}
  for i, j, s, n, *_ in games:
    a = agg.setdefault((pos[i], pos[j]), [0.5, 1.0])
    a[0] += s
    a[1] += n
  r = np.zeros(len(players))
  for _ in range(100):
    g, h = np.zeros_like(r), np.zeros((len(r), len(r)))
    for (i, j), (s, n) in agg.items():
      p = 1 / (1 + math.exp(r[j] - r[i]))
      g[i] += s - n * p
      g[j] -= s - n * p
      w = n * p * (1 - p)
      h[i, i] -= w
      h[j, j] -= w
      h[i, j] += w
      h[j, i] += w
    if len(r) < 2:
      break
    step = np.linalg.solve(h[1:, 1:] - 1e-9 * np.eye(len(r) - 1), g[1:])
    r[1:] -= step
    if np.abs(step).max() < 1e-6:
      break
  return r * ELO


def role_matrix(players: list[int], games: list[list]) -> np.ndarray:
  """Square over checkpoints, the newer one of each pair as subject: below the diagonal (row newer)
  the row's tag rate as pursuer vs the column; above it (column newer) the column's not-tagged
  rate as evader vs the row."""
  pos = {p: a for a, p in enumerate(players)}
  m = np.full((len(players),) * 2, np.nan)
  for i, j, _, _, n_p, tags, n_e, tagged in games:  # i: the newer checkpoint.
    a, b = pos[i], pos[j]
    m[a, b] = tags / n_p if n_p else np.nan
    m[b, a] = 1 - tagged / n_e if n_e else np.nan
  return m


def plot_roles(state: dict, out: str, title: str | None, pdf: bool = True, transparent: bool = False,
               dpi: int = 130) -> str:
  """title None: no title. transparent: the figure background (outside the axes) is transparent; the axes keep
  their white face, so the diagonal and empty cells look as they do on a white page."""
  players = state["players"]
  m = role_matrix(players, state["games"])
  low = np.tril(np.ones_like(m, dtype=bool), -1)
  fig, (ax, cp, ce) = plt.subplots(1, 3, figsize=(14, 11), gridspec_kw={"width_ratios": (1, 0.03, 0.03), "wspace": 0.25})
  ip = ax.imshow(np.ma.masked_where(~low | np.isnan(m), m), cmap="Reds", vmin=0, vmax=1)
  ie = ax.imshow(np.ma.masked_where(low | np.isnan(m), m), cmap="Blues", vmin=0, vmax=1)
  if len(players) <= 25:
    for (a, b), v in np.ndenumerate(m):
      if not np.isnan(v):
        ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w" if v > 0.6 else "k")
  step = max(1, len(players) // 25)
  ticks = list(range(0, len(players), step))
  ax.set_xticks(ticks, [str(players[t]) for t in ticks], rotation=60, ha="right")
  ax.set_yticks(ticks, [str(players[t]) for t in ticks])
  ax.set_xlabel("checkpoint (iteration)")
  ax.set_ylabel("checkpoint (iteration)")
  fig.colorbar(ip, cax=cp, label="below diagonal: row's pursuer win rate (tags the column)")
  fig.colorbar(ie, cax=ce, label="above diagonal: column's evader win rate (not tagged by the row)")
  n = int(np.median([g[4] for g in state["games"]])) if state["games"] else 0
  if title:
    ax.set_title(f"{title}: newer checkpoint of each pair vs the older, by role\n"
                 f"deterministic head-to-head, ~{n} games per pairing and role; evader 'win' = not tagged (timeouts are draws)",
                 loc="left", fontsize=11)
  if transparent:
    fig.patch.set_alpha(0)
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  fig.savefig(out, dpi=dpi, bbox_inches="tight")
  if pdf:
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
  plt.close(fig)
  print(f"saved {out}")
  return out


def score_matrix(players: list[int], games: list[list]) -> np.ndarray:
  """Mean score of row vs column (nan where they never met)."""
  pos = {p: a for a, p in enumerate(players)}
  s, n = np.zeros((len(players),) * 2), np.zeros((len(players),) * 2)
  for i, j, sc, ng, *_ in games:
    a, b = pos[i], pos[j]
    s[a, b] += sc
    n[a, b] += ng
    s[b, a] += ng - sc
    n[b, a] += ng
  with np.errstate(invalid="ignore"):
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def plot(state: dict, out: str, title: str, pdf: bool = True) -> str:
  players, games = state["players"], state["games"]
  elo, m = fit(players, games), score_matrix(players, games)
  fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={"width_ratios": (1, 1.15)})
  a1.plot(players, elo, "o-", color="#333", ms=4)
  a1.axhline(0, color="0.7", lw=0.8)
  a1.set_xlabel("checkpoint (training iteration)")
  a1.set_ylabel("Elo (model_0 = 0)")
  a1.set_title("Elo from head-to-head games between checkpoints", fontweight="bold", loc="left")
  a1.grid(alpha=0.3)
  im = a2.imshow(np.ma.masked_invalid(m), cmap="RdBu", norm=TwoSlopeNorm(0.5, 0.0, 1.0))
  step = max(1, len(players) // 20)
  ticks = list(range(0, len(players), step))
  a2.set_xticks(ticks, [str(players[t]) for t in ticks], rotation=60, ha="right")
  a2.set_yticks(ticks, [str(players[t]) for t in ticks])
  a2.set_xlabel("opponent checkpoint")
  a2.set_ylabel("checkpoint")
  a2.set_title("Score of row vs column (1 = row always tags first, 0.5 = even)", fontweight="bold", loc="left")
  fig.colorbar(im, ax=a2, fraction=0.04, pad=0.02)
  n = int(np.median([g[3] for g in games])) if games else 0
  fig.suptitle(f"{title}: deterministic actions, ~{n} games per pairing, roles 50/50", x=0.01, ha="left")
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  fig.savefig(out, dpi=130)
  if pdf:
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
  plt.close(fig)
  print(f"saved {out}")
  return out


class Tournament:
  """A run's elo.json: ``add(env, ckpt)`` plays the new checkpoint against all earlier ones."""

  def __init__(self, run: str):
    self.path = os.path.join(run, "plots", "elo.json")
    os.makedirs(os.path.dirname(self.path), exist_ok=True)
    self.state = json.load(open(self.path)) if os.path.exists(self.path) else {"players": [], "games": []}
    self.run = run

  def _play(self, env, it: int, prev: list[int], seed: int):
    res = head_to_head(env, os.path.join(self.run, f"model_{it}.pt"), [os.path.join(self.run, f"model_{p}.pt") for p in prev], seed)
    self.state["games"] += [[it, p, *r] for p, r in zip(prev, res)]
    return res

  def add(self, env, ckpt: str, seed: int = 0) -> dict[str, float]:
    it = ckpt_iter(ckpt)
    if it in self.state["players"]:
      return {}
    if any(len(g) < 8 for g in self.state["games"]):  # Recorded before the role split: replay (same seed, same games).
      players, self.state["games"] = self.state["players"], []
      for a, p in enumerate(players[1:], 1):
        self._play(env, p, players[:a], seed)
    prev = self.state["players"]
    if prev:
      res = self._play(env, it, prev, seed)
    self.state["players"] = prev + [it]
    json.dump(self.state, open(self.path, "w"))
    elo = fit(self.state["players"], self.state["games"])
    row = {"elo/latest": float(elo[-1])}
    if prev:
      m = {p: r[0] / r[1] for p, r in zip(prev, res)}
      row.update({"elo/score_vs_previous": m[prev[-1]], "elo/score_vs_model_0": m[prev[0]],
                  "elo/score_vs_all_earlier": float(np.mean(list(m.values())))})
    return row


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--device", default=None)
  a = ap.parse_args()
  t = Tournament(a.run)
  env = plot_states.load_env(a.run, a.num_envs, a.device)
  for ckpt in sorted(glob.glob(os.path.join(a.run, "model_*.pt")), key=ckpt_iter):
    print(ckpt_iter(ckpt), t.add(env, ckpt))
  plot(t.state, os.path.join(a.run, "plots", "elo.png"), os.path.basename(os.path.abspath(a.run)))
  plot_roles(t.state, os.path.join(a.run, "plots", "role_matrix.png"), os.path.basename(os.path.abspath(a.run)))


if __name__ == "__main__":
  main()
