"""Host-side opponent league: prioritized fictitious self-play (PFSP, as in AlphaStar) over
frozen learner snapshots and the scripted stage-1 baseline, plus the env partition between
symmetric self-play and league groups. No simulator code; testable on CPU."""

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

SCRIPTED = "scripted"
SCRIPTED_ID = "scripted/stage1"


@dataclass(frozen=True)
class Opponent:
  """A frozen or scripted opponent: maps opponent obs to opponent actions."""

  opponent_id: str
  policy: Callable[[torch.Tensor], torch.Tensor]
  category: str = SCRIPTED  # scripted (never evicted) | learner | historical
  path: str | None = None  # rsl-rl-format checkpoint, so a resumed run can reload it.


@dataclass(frozen=True)
class LeagueCfg:
  """Hydra group ``league``. The defaults reproduce scripted-only play without self-play."""

  selfplay_frac: float = 0.0  # Fraction of envs where the learner plays both seats.
  sampling: str = "pfsp"  # pfsp: weight (1 - p)^pfsp_power, p = learner's EMA win rate | uniform | recent
  pfsp_power: float = 2.0
  win_ema: float = 0.05  # EMA rate per episode of the learner's win rate vs each member.
  init_win: float = 0.5  # Win rate assumed for an unseen member.
  min_weight: float = 0.02  # Floor on every final probability (capped at 1 / pool size).
  recent_decay: float = 0.9  # recent: the k-th newest member has weight recent_decay^k.
  snapshot_every: int = 0  # Iterations between learner snapshots (0: off).
  max_size: int = 32  # Pool limit: evicts the oldest non-scripted member.
  groups: int = 1  # League env groups, one opponent each.
  resample_every: int = 1200  # Env steps between redraws of each group's opponent.
  scripted_weight_start: float = 1.0  # Scripted weight multiplier, linear from start to end
  scripted_weight_end: float = 1.0  # over scripted_decay_iters iterations (1 -> 1: no effect).
  scripted_decay_iters: int = 1
  init_pool: list[str] = field(default_factory=list)  # Checkpoints preloaded as historical members.
  log_all_members: bool = False  # Log every member's league/p and league/win (else scripted + top 3).
  # Adaptive role sampling: once per iteration, the probability that a league env's learner plays the
  # evader moves by evader_rate_step toward balance: up if the learner-evader win rate (league and
  # self-play games finished that iteration) is below 50%, down if above. 0: off (50/50).
  evader_rate_step: float = 0.0
  evader_rate_min: float = 0.05
  evader_rate_max: float = 0.95


@dataclass(frozen=True)
class Partition:
  """Contiguous env slices: self-play envs first, then one slice per league group."""

  selfplay: slice
  groups: tuple[slice, ...]


class League:
  """Opponent pool with seeded PFSP / uniform / recent sampling.

  ``load(path) -> policy`` rebuilds a checkpoint's policy (e.g. ``load_policy`` on the run
  device); it is needed for ``init_pool``, state-dict snapshots and ``load_state_dict``.
  Snapshots are saved to ``snapshot_dir`` (as ``{"actor_state_dict": ...}``) when given.
  """

  def __init__(self, cfg: LeagueCfg, seed: int = 0, load: Callable[[str], Callable] | None = None,
               snapshot_dir: str | None = None) -> None:
    if cfg.sampling not in ("pfsp", "uniform", "recent"):
      raise ValueError(f"unknown league sampling {cfg.sampling!r}")
    self.cfg, self.load, self.snapshot_dir = cfg, load, snapshot_dir
    self.members: list[Opponent] = []
    self.win: dict[str, float] = {}  # Learner's EMA win rate (draw = 0.5) vs each member.
    self.games: dict[str, int] = {}
    self.rng = torch.Generator().manual_seed(seed)
    self.partition, self.current = Partition(slice(0, 0), ()), []
    for path in cfg.init_pool:
      self.add(Opponent(f"historical/{path}", load(path), "historical", path))

  def add(self, opponent: Opponent) -> None:
    """Append a member; past ``max_size`` the oldest non-scripted member is evicted."""
    if opponent.opponent_id in self.win:
      raise ValueError(f"duplicate opponent {opponent.opponent_id}")
    self.members.append(opponent)
    self.win[opponent.opponent_id], self.games[opponent.opponent_id] = self.cfg.init_win, 0
    evictable = [o for o in self.members if o.category != SCRIPTED]
    if len(self.members) > self.cfg.max_size and evictable:
      self.members.remove(evictable[0])
      del self.win[evictable[0].opponent_id], self.games[evictable[0].opponent_id]

  def snapshot(self, actor: torch.nn.Module | dict, iteration: int) -> Opponent:
    """Freeze a copy of the learner's actor (rsl-rl MLPModel, any module, or a state dict)."""
    sd, path = actor.state_dict() if isinstance(actor, torch.nn.Module) else actor, None
    if self.snapshot_dir:
      os.makedirs(self.snapshot_dir, exist_ok=True)
      path = os.path.join(self.snapshot_dir, f"learner_{iteration}.pt")
      torch.save({"actor_state_dict": sd, "iter": iteration}, path)
    if isinstance(actor, torch.nn.Module):  # as_jit: deep copy taking the obs tensor directly.
      policy = (actor.as_jit() if hasattr(actor, "as_jit") else copy.deepcopy(actor)).eval().requires_grad_(False)
    else:
      assert path and self.load, "state-dict snapshots need snapshot_dir and load"
      policy = self.load(path)
    opponent = Opponent(f"learner/{iteration}", policy, "learner", path)
    self.add(opponent)
    return opponent

  def scripted_weight(self, iteration: int) -> float:
    c = self.cfg
    t = min(max(iteration / max(c.scripted_decay_iters, 1), 0.0), 1.0)
    return c.scripted_weight_start + t * (c.scripted_weight_end - c.scripted_weight_start)

  def probs(self, iteration: int) -> dict[str, float]:
    """Sampling weights, times the scripted multiplier, normalized, then floored at min_weight."""
    c, n = self.cfg, len(self.members)
    if c.sampling == "pfsp":
      win = torch.tensor([self.win[o.opponent_id] for o in self.members], dtype=torch.float64)
      w = (1 - win).clamp_min(0) ** c.pfsp_power
    elif c.sampling == "uniform":
      w = torch.ones(n, dtype=torch.float64)
    else:  # recent: members are in insertion order, newest last.
      w = c.recent_decay ** torch.arange(n - 1, -1, -1, dtype=torch.float64)
    w = w * torch.tensor([self.scripted_weight(iteration) if o.category == SCRIPTED else 1.0 for o in self.members],
                         dtype=torch.float64)
    if w.sum() <= 0:  # Every member beaten: play them all equally.
      w = torch.ones(n, dtype=torch.float64)
    p, floor, low = w / w.sum(), min(c.min_weight, 1 / n), torch.zeros(n, dtype=torch.bool)
    while (new := (p < floor) & ~low).any():  # Lift members to the floor; rescale the rest.
      low |= new
      p = torch.where(low, floor, w * (1 - floor * low.sum().item()) / w[~low].sum())
    return dict(zip([o.opponent_id for o in self.members], p.tolist()))

  def sample(self, iteration: int) -> Opponent:
    p = torch.tensor(list(self.probs(iteration).values()))
    return self.members[torch.multinomial(p, 1, generator=self.rng).item()]

  def assign(self, num_envs: int) -> Partition:
    """Split envs: round(selfplay_frac * num_envs) self-play envs, the rest into up to
    ``groups`` contiguous league groups whose sizes differ by at most one."""
    n_sp = round(num_envs * self.cfg.selfplay_frac)
    n, k = num_envs - n_sp, min(self.cfg.groups, num_envs - n_sp)
    size, extra = divmod(n, k) if k else (0, 0)
    starts = [n_sp + i * size + min(i, extra) for i in range(k + 1)]
    self.partition = Partition(slice(0, n_sp), tuple(slice(a, b) for a, b in zip(starts, starts[1:])))
    return self.partition

  def resample_groups(self, iteration: int) -> list[Opponent]:
    """Draw a fresh opponent for each group of the last ``assign``."""
    self.current = [self.sample(iteration) for _ in self.partition.groups]
    return self.current

  def record(self, opponent_id: str, wins) -> None:
    """Fold episode outcomes vs a member (+1 learner win, -1 loss, 0 draw = half a win) into
    its EMA: n outcomes count as n EMA steps toward their mean. Evicted members are ignored."""
    w = torch.as_tensor(wins, dtype=torch.float64).flatten().cpu()
    if opponent_id not in self.win or not len(w):
      return
    rate = 1 - (1 - self.cfg.win_ema) ** len(w)
    self.win[opponent_id] += rate * ((w.mean().item() + 1) / 2 - self.win[opponent_id])
    self.games[opponent_id] += len(w)

  def stats(self, iteration: int) -> dict[str, float]:
    """Flat scalars for tensorboard. share/<category> is the expected fraction of all envs."""
    p, sp = self.probs(iteration), self.cfg.selfplay_frac
    out = {"league/size": len(self.members), "league/share/selfplay": sp,
           "league/scripted_weight": self.scripted_weight(iteration),
           "league/p_scripted": sum(p[o.opponent_id] for o in self.members if o.category == SCRIPTED)}
    for o in self.members:
      key = f"league/share/{o.category}"
      out[key] = out.get(key, 0.0) + (1 - sp) * p[o.opponent_id]
      out[f"league/p/{o.opponent_id}"], out[f"league/win/{o.opponent_id}"] = p[o.opponent_id], self.win[o.opponent_id]
    return out

  def state_dict(self) -> dict:
    return {"members": [(o.opponent_id, o.category, o.path) for o in self.members],
            "win": dict(self.win), "games": dict(self.games), "rng": self.rng.get_state()}

  def load_state_dict(self, sd: dict) -> None:
    """Restore the pool, stats and RNG. Members already added (e.g. the scripted one) are
    kept; saved ones are reloaded from their paths; unsaved snapshots are dropped."""
    have = {o.opponent_id: o for o in self.members}
    saved = [have.get(i) or Opponent(i, self.load(path), cat, path) for i, cat, path in sd["members"] if i in have or path]
    self.members = saved + [o for o in have.values() if o not in saved]
    ids = {o.opponent_id for o in self.members}
    self.win = {i: sd["win"].get(i, self.win.get(i, self.cfg.init_win)) for i in ids}
    self.games = {i: sd["games"].get(i, self.games.get(i, 0)) for i in ids}
    self.rng.set_state(sd["rng"])


class FixedLeague:
  """One opponent, behind OpponentEnv's pre-league ``sample(seed)`` API (scripted play, eval, checks)."""

  def __init__(self, opponent: Opponent) -> None:
    self.opponent = opponent

  def sample(self, seed: int = 0) -> Opponent:
    return self.opponent
