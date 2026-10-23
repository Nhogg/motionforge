"""Host-side opponent: rsl-rl sees only the learner; the league drives the opponent."""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mofo.league import SCRIPTED_ID, League, Partition
from mofo.selfplay import SeatRewards
from mofo.tasks.tag import ACTION, TAG_DIM, outcome, role, tag_rules


def load_policy(path: str, device: str) -> torch.nn.Module:
  """Deterministic actor (normalizer + MLP) from an rsl-rl checkpoint."""
  sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
  ws = [v for k, v in sd.items() if k.startswith("mlp.") and k.endswith(".weight")]
  obs = TensorDict({"actor": torch.zeros(1, ws[0].shape[1])}, batch_size=[1])
  model = MLPModel(
    obs, {"actor": ["actor"]}, "actor", ws[-1].shape[0], [w.shape[0] for w in ws[:-1]],
    obs_normalization="obs_normalizer._mean" in sd,
  )
  model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("distribution.")})
  return model.as_jit().to(device).eval()


class ScriptedPolicy:
  """Stage-1 velocity policy driven by a scripted chase/flee command.

  Pursuer: command toward the other robot. Evader: away from it, bent toward the arena
  center near the wall. Clipped to the stage-1 command ranges. With ``policy=None`` it
  outputs zero actions (hold the default pose), a placeholder until stage 1 exists.
  """

  def __init__(self, env: ManagerBasedRlEnv, policy=None, me="opponent", margin=1.5):
    self.env, self.policy, self.me, self.margin = env, policy, me, margin
    self.other = "robot" if me == "opponent" else "opponent"
    self.role_sign = -1.0 if me == "opponent" else 1.0
    self.num_actions = env.action_manager.get_term(ACTION[me]).action_dim
    r = unitree_g1_flat_env_cfg().commands["twist"].ranges
    lo_hi = list(zip(r.lin_vel_x, r.lin_vel_y, r.ang_vel_z))
    self.lo, self.hi = (torch.tensor(v, device=env.device) for v in lo_hi)
    # Turn side for a target exactly behind (atan2 at +-pi, e.g. the facing spawn): a per-env
    # random sign redrawn each episode. A private generator leaves the env's RNG untouched.
    self.gen = torch.Generator(env.device).manual_seed(0)
    self.side = torch.ones(env.num_envs, device=env.device)
    self.drawn = torch.full((env.num_envs,), -1, device=env.device)  # Step counter of the last draw.

  def command(self, ids: slice = slice(None)) -> torch.Tensor:
    """Commands for every env; ``ids`` (the envs this member drives) only scopes the side draws."""
    env = self.env
    a, b = env.scene[self.me].data, env.scene[self.other].data
    to = b.root_link_pos_w - a.root_link_pos_w
    to[:, 2] = 0
    to = to / to.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    xy = a.root_link_pos_w - env.scene.env_origins
    xy[:, 2] = 0
    edge = ((xy[:, :2].abs().max(-1).values - tag_rules(env).arena_half_extent + self.margin) / self.margin).clamp(0, 1)
    home = -xy / xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    flee = -to + 2 * edge[:, None] * home
    pursue = (self.role_sign * role(env) > 0)[:, None]
    v = quat_apply_inverse(yaw_quat(a.root_link_quat_w), torch.where(pursue, to, flee))
    heading = torch.atan2(v[:, 1], v[:, 0])
    idx = torch.arange(env.num_envs, device=env.device)[ids]
    new = idx[(env.episode_length_buf[idx] == 0) & (self.drawn[idx] != env.common_step_counter)]  # Once per episode.
    self.drawn[new] = env.common_step_counter
    self.side[new] = torch.randint(2, (len(new),), generator=self.gen, device=env.device) * 2.0 - 1
    heading = torch.where(heading.abs() > math.pi - 1e-3, self.side * math.pi, heading)
    v = v[:, :2] / v[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cmd = torch.cat([v * self.hi[:2], 2.0 * heading[:, None]], -1)
    return torch.maximum(torch.minimum(cmd, self.hi), self.lo)

  def __call__(self, obs: torch.Tensor, ids: slice = slice(None)) -> torch.Tensor:
    """obs: this robot's tag obs group (stage-1 proprio terms, then tag features) for envs ``ids``."""
    if self.policy is None:
      return torch.zeros(obs.shape[0], self.num_actions, device=obs.device)
    return self.policy(torch.cat([obs[:, :-TAG_DIM], self.command(ids)[ids]], -1))


def _league_log(stats: dict[str, float], top: int = 3) -> dict[str, float]:
  """League stats minus the per-member long tail: keep the scripted member and the ``top`` likeliest."""
  members = sorted((k for k in stats if k.startswith("league/p/") and k != f"league/p/{SCRIPTED_ID}"), key=stats.get, reverse=True)
  keep = {f"league/p/{SCRIPTED_ID}", f"league/win/{SCRIPTED_ID}", *members[:top], *(k.replace("/p/", "/win/") for k in members[:top])}
  return {k: v for k, v in stats.items() if k in keep or not k.startswith(("league/p/", "league/win/"))}


class OpponentEnv(RslRlVecEnvWrapper):
  """mjlab's rsl-rl wrapper, exposing only the learner's seats.

  Envs are partitioned: self-play envs first, where the learner plays both seats, then
  opponent groups. rsl-rl sees N + n_sp envs: the N robot seats, then the n_sp self-play
  opponent seats, with per-seat obs (``opponent``, ``opponent_critic``), actions, rewards
  (``SeatRewards``) and duplicated dones. In group envs only the robot seat learns.

  ``league``: a ``League`` pool (training) partitions with ``league.assign``, redraws one
  member per group every ``cfg.resample_every`` steps, records the learner's episode outcome
  vs each member, freezes ``self.learner`` (set by the trainer) every ``cfg.snapshot_every``
  iterations, and logs its stats once per iteration. A ``FixedLeague`` (eval, checks) drives
  every group env with one opponent, and ``selfplay_frac`` sets the self-play envs.
  """

  def __init__(self, env: ManagerBasedRlEnv, league, resample_every: int = 24 * 50, selfplay_frac: float = 0.0, steps_per_iter: int = 24):
    self.league, self.steps, self.steps_per_iter, self.learner = league, 0, steps_per_iter, None
    super().__init__(env)
    N = env.num_envs
    self.pool = isinstance(league, League)
    if self.pool:
      self.part, self.resample_every = league.assign(N), league.cfg.resample_every
    else:
      n = round(selfplay_frac * N)
      self.part, self.resample_every = Partition(slice(0, n), (slice(n, N),) if n < N else ()), resample_every
    self.n_sp = self.part.selfplay.stop
    self.num_envs += self.n_sp
    self.num_actions = env.action_manager.get_term(ACTION["robot"]).action_dim
    self.seats = SeatRewards(env) if self.n_sp else None
    self.evader_games = torch.zeros(2, 2, device=env.device)  # [league, self-play] x [games, evader wins], this iteration.

  def _learner(self, obs: dict) -> TensorDict:
    """Learner seats' obs: every robot seat, then the self-play opponent seats."""
    n = self.n_sp
    if n:
      obs = {"actor": torch.cat([obs["actor"], obs["opponent"][:n]]), "critic": torch.cat([obs["critic"], obs["opponent_critic"][:n]])}
    return TensorDict(obs, batch_size=[self.num_envs]).exclude("opponent")

  def get_observations(self) -> TensorDict:
    return self._learner(self.unwrapped.observation_manager.compute())

  def reset(self) -> tuple[TensorDict, dict]:
    obs, extras = self.env.reset()
    return self._learner(obs), extras

  def _league_step(self) -> dict:
    """Snapshot the learner (iteration boundaries), redraw opponents (resample steps), and
    once per iteration log the league stats and the actual env share per opponent category."""
    it, log, boundary = self.steps // self.steps_per_iter, {}, self.steps % self.steps_per_iter == 0
    c = self.league.cfg if self.pool else None
    if self.pool and boundary and c.snapshot_every and it and it % c.snapshot_every == 0 and self.learner is not None:
      self.league.snapshot(self.learner, it)
    if self.steps % self.resample_every == 0:
      self.opponents = (
        self.league.resample_groups(it) if self.pool else [self.league.sample(seed=self.steps // self.resample_every)] * len(self.part.groups)
      )
    if self.pool and boundary:
      N, envs = self.unwrapped.num_envs, {"selfplay": self.n_sp}
      for o, s in zip(self.opponents, self.part.groups):
        envs[o.category] = envs.get(o.category, 0) + s.stop - s.start
      stats = self.league.stats(it)
      log = {**(stats if c.log_all_members else _league_log(stats)), **{f"league/envs/{k}": v / N for k, v in envs.items()},
             **self._adapt_evader_rate(c)}
    self.steps += 1
    return log

  def _adapt_evader_rate(self, c) -> dict:
    """Once per iteration: the learner-evader win rate over the games finished since the last call
    and, if c.evader_rate_step, nudge the role term's p_evader toward a 50% evader win rate."""
    g = self.evader_games.tolist()
    self.evader_games.zero_()
    term, log = self.unwrapped.command_manager.get_term("role"), {}
    for (games, wins), k in zip(g, ("league", "selfplay")):
      if games:
        log[f"league/evader_win_rate/{k}"] = wins / games
    games, wins = g[0][0] + g[1][0], g[0][1] + g[1][1]
    if games:
      rate = wins / games
      log["league/evader_win_rate"] = rate
      if c.evader_rate_step and rate != 0.5:
        step = c.evader_rate_step if rate < 0.5 else -c.evader_rate_step
        term.p_evader = min(max(term.p_evader + step, c.evader_rate_min), c.evader_rate_max)
    log["league/p_evader"] = term.p_evader
    return log

  def step(self, actions: torch.Tensor):
    """As RslRlVecEnvWrapper.step (no clip_actions), with the opponents' actions appended."""
    log = self._league_step()
    env, n, N = self.unwrapped, self.n_sp, self.unwrapped.num_envs
    obs_opp, learner_role = env.obs_buf["opponent"], role(env).clone()
    with torch.no_grad():  # Self-play seats act from the learner; each group from its opponent.
      opp = [actions[N:]] + [
        o.policy(obs_opp[s], s) if isinstance(o.policy, ScriptedPolicy) else o.policy(obs_opp[s])
        for o, s in zip(self.opponents, self.part.groups)
      ]
    obs, rew, terminated, truncated, extras = self.env.step(torch.cat([actions[:N], torch.cat(opp)], -1))
    if self.pool:
      done, out = terminated | truncated, outcome(env, learner_role)
      wins: dict[str, list] = {}
      for o, s in zip(self.opponents, self.part.groups):
        if done[s].any():
          self.league.record(o.opponent_id, out[s][done[s]])
          wins.setdefault(o.category, []).append((out[s][done[s]] > 0).float())
      sp = done[:n]
      if sp.any():  # Self-play: how often the pursuer seat wins.
        wins["selfplay_pursuer"] = [(out[:n][sp] * learner_role[:n][sp] > 0).float()]
      # Learner-evader games: league envs where the learner was the evader (its win: out > 0), and
      # every self-play game (one learner seat is the evader; it won if the pursuer seat lost).
      ev = done[n:] & (learner_role[n:] < 0)
      self.evader_games[0, 0] += ev.sum()
      self.evader_games[0, 1] += (ev & (out[n:] > 0)).sum()
      self.evader_games[1, 0] += sp.sum()
      self.evader_games[1, 1] += (sp & (out[:n] * learner_role[:n] < 0)).sum()
      log.update({f"league/win_rate/{k}": torch.cat(v) for k, v in wins.items()})
    if n:
      robot, other = self.seats(env)
      rew = torch.cat([robot, other[:n]])
      terminated, truncated = (torch.cat([x, x[:n]]) for x in (terminated, truncated))
      log["Seat/robot_step_reward"], log["Seat/opponent_step_reward"] = robot[:n].mean(), other[:n].mean()
    extras["log"].update(log)
    obs = self._reset_nonfinite(obs, rew, terminated, extras)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return self._learner(obs), rew, (terminated | truncated).to(dtype=torch.long), extras

  def _reset_nonfinite(self, obs: dict, rew: torch.Tensor, terminated: torch.Tensor, extras: dict) -> dict:
    """Reset worlds whose observations went NaN/Inf (a physics blow-up that the nan_reset
    termination, which checks only the raw sim state before observations are computed, can miss):
    their seats end the episode with zero reward. In place on ``rew`` / ``terminated`` (seat order);
    returns the observations. A no-op unless something is non-finite."""
    env, N = self.unwrapped, self.unwrapped.num_envs
    bad = torch.zeros(N, dtype=torch.bool, device=env.device)
    for v in obs.values():
      bad |= ~torch.isfinite(v).flatten(1).all(-1)
    seats = torch.cat([bad, bad[: self.n_sp]])
    bad_rew = ~torch.isfinite(rew)
    if not bad.any() and not bad_rew.any():
      return obs
    rew[seats | bad_rew] = 0.0
    if bad.any():
      log = extras["log"]
      obs, _ = env.reset(env_ids=bad.nonzero()[:, 0])  # Replaces extras["log"]; this step's entries win.
      extras["log"] = {**extras["log"], **log, "Episode_Termination/nonfinite_obs_reset": bad.sum().item()}
      obs = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in obs.items()}  # If a reset itself is bad.
      terminated[seats] = True
    return obs
