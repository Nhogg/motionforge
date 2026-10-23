"""Learner-vs-pool matrix from a tag run's W&B league logs: the learner's EMA win rate against
each pool member (left) and the PFSP probability of drawing it (right), one row per snapshot.

Row @k is the log at iteration k-1, just before snapshot k joins the pool; the last row is the
run's latest iteration. Runs trained with league.log_all_members=true log every member each
iteration. Older runs log only scripted + the top 3: --ffill carries each member's last logged
win rate forward and recomputes PFSP from it (the method behind the original L_s1 figure), so
most cells are stale there.

  uv run scripts/plot_league_matrix.py luc-ssl/mofo/tag_scratch_lim-L_s1_repro2 [--every 250] [--until 4366] [--out OUT.png]
  uv run scripts/plot_league_matrix.py --tb logs/mofo/tag/tag_scratch_lim/L_s0   # the run's local tensorboard log
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from mofo.league import SCRIPTED, SCRIPTED_ID, League, LeagueCfg, Opponent

WIN, P = "league/win/", "league/p/"


def load_history(path: str) -> tuple["wandb.apis.public.Run", dict[int, dict[str, float]]]:
  """iteration -> {league key: value}, keeping only the league/win and league/p keys."""
  import wandb

  run = wandb.Api(timeout=300).run(path)
  hist: dict[int, dict[str, float]] = {}
  for row in run.scan_history(page_size=2000):
    it = row.get("iteration")
    vals = {k: v for k, v in row.items() if k.startswith((WIN, P)) and v is not None}
    if it is not None and vals:
      hist.setdefault(int(it), {}).update(vals)
  return run, hist


def tb_history(ea) -> dict[int, dict[str, float]]:
  """As load_history, from a tensorboard EventAccumulator on the run dir (call ea.Reload() first)."""
  hist: dict[int, dict[str, float]] = {}
  for tag in ea.Tags()["scalars"]:
    if tag.startswith((WIN, P)):
      for e in ea.Scalars(tag):
        hist.setdefault(e.step, {})[tag] = e.value
  return hist


def league_cfg(league: dict) -> LeagueCfg:
  return LeagueCfg(**{k: v for k, v in league.items() if k in LeagueCfg.__dataclass_fields__})


def member_iter(mid: str) -> int:
  return -1 if mid == SCRIPTED_ID else int(re.fullmatch(r"learner/(\d+)", mid).group(1))


def label(mid: str) -> str:
  return "scripted" if mid == SCRIPTED_ID else f"L{member_iter(mid)}"


def pfsp(win: dict[str, float], cfg: LeagueCfg) -> dict[str, float]:
  """League.probs for these EMA win rates (scripted multiplier 1, as in tag_scratch_lim)."""
  lg = League(cfg)
  for mid, w in sorted(win.items(), key=lambda kv: member_iter(kv[0])):
    lg.add(Opponent(mid, None, SCRIPTED if mid == SCRIPTED_ID else "learner"))
    lg.win[mid] = w
  return lg.probs(0)


def matrices(hist, every: int, until: int | None, ffill: bool, cfg: LeagueCfg):
  last = max(it for it in hist if until is None or it <= until)
  rows = [k for k in range(every, last + 1, every)] + ([last] if last % every else [])
  seen: dict[str, float] = {}  # --ffill: latest win rate per member.
  win_rows, p_rows = [], []
  for k in rows:
    it = k - 1 if k != last else k
    if ffill:
      for i in sorted(i for i in hist if i <= it):
        seen.update({key[len(WIN):]: v for key, v in hist[i].items() if key.startswith(WIN)})
      # Members in the pool at `it`: scripted plus every snapshot taken by then (no eviction before 32).
      win = {m: v for m, v in seen.items() if member_iter(m) <= it}
      p = pfsp(win, cfg)
    else:
      row = hist.get(it, {})
      win = {key[len(WIN):]: v for key, v in row.items() if key.startswith(WIN)}
      p = {key[len(P):]: v for key, v in row.items() if key.startswith(P)}
    win_rows.append(win)
    p_rows.append(p)
  members = sorted({m for r in win_rows for m in r}, key=member_iter)
  grid = lambda rs: np.array([[r.get(m, np.nan) for m in members] for r in rs])
  return rows, members, grid(win_rows), grid(p_rows)


def draw(ax, data, rows, members, cmap, norm, fmt, title, fig):
  im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, aspect="auto")
  ax.set_xticks(range(len(members)), [label(m) for m in members], rotation=60, ha="right")
  ax.set_yticks(range(len(rows)), [f"@{k}" for k in rows])
  ax.set_xlabel("pool member (opponent)")
  ax.set_ylabel("learner at iteration")
  ax.set_title(title, fontweight="bold", loc="left")
  if len(members) <= 25:
    for (i, j), v in np.ndenumerate(data):
      if not np.isnan(v):
        ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7)
  fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def plot(hist, cfg: LeagueCfg, out: str, title: str, every: int = 250, until: int | None = None, ffill: bool = False,
         pdf: bool = True) -> str:
  rows, members, win, p = matrices(hist, every, until, ffill, cfg)
  fig, axes = plt.subplots(1, 2, figsize=(20, 9.6))
  draw(axes[0], win, rows, members, "RdBu", TwoSlopeNorm(0.5, 0.0, 1.0), lambda v: f"{v:.2f}",
       "EMA win rate p: learner (row, at iteration) vs member (col)", fig)
  draw(axes[1], p, rows, members, "Blues", plt.Normalize(0.0, 1.0), lambda v: f"{v:.0%}",
       "→ PFSP probability of drawing that member", fig)
  fig.suptitle(f"{title} — learner-vs-pool matrix, sampled just before each snapshot joins"
               + (" (last-logged values carried forward)" if ffill else ""), x=0.01, ha="left")
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  fig.savefig(out, dpi=150 if pdf else 110)
  if pdf:
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
  plt.close(fig)
  print(f"saved {out} ({len(rows)} rows x {len(members)} members)")
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run", nargs="?", help="W&B run path, entity/project/id")
  ap.add_argument("--tb", default=None, help="Read the league logs from this run dir's tensorboard file instead of W&B.")
  ap.add_argument("--every", type=int, default=250, help="Row stride in iterations (the snapshot interval).")
  ap.add_argument("--until", type=int, default=None, help="Last iteration to include.")
  ap.add_argument("--ffill", action="store_true", help="Carry each member's last logged win rate forward.")
  ap.add_argument("--out", default=None, help="Default: logs/plots/league_matrix_<run id>.png")
  a = ap.parse_args()

  if a.tb:
    import yaml
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    hist = tb_history(EventAccumulator(a.tb, size_guidance={"scalars": 0}).Reload())
    rid = f"{os.path.basename(os.path.dirname(os.path.abspath(a.tb)))}-{os.path.basename(os.path.abspath(a.tb))}"
    cfg, title = league_cfg(yaml.safe_load(open(os.path.join(a.tb, ".hydra", "config.yaml")))["league"]), rid.replace("-", " / ", 1)
  else:
    run, hist = load_history(a.run)
    rid, cfg, title = run.id, league_cfg(run.config["league"]), f"{run.group or ''} / {run.name}"
  plot(hist, cfg, a.out or f"logs/plots/league_matrix_{rid}{'_ffill' if a.ffill else ''}.png", title, a.every, a.until, a.ffill)

if __name__ == "__main__":
  main()
