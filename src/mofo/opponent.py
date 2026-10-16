"""Host-side opponent: rsl-rl sees only the learner; the league drives the opponent."""

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

  def command(self) -> torch.Tensor:
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
    v = v[:, :2] / v[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cmd = torch.cat([v * self.hi[:2], 2.0 * heading[:, None]], -1)
    return torch.maximum(torch.minimum(cmd, self.hi), self.lo)

  def __call__(self, obs: torch.Tensor, ids: slice = slice(None)) -> torch.Tensor:
    """obs: this robot's tag obs group (stage-1 proprio terms, then tag features) for envs ``ids``."""
    if self.policy is None:
      return torch.zeros(obs.shape[0], self.num_actions, device=obs.device)
    return self.policy(torch.cat([obs[:, :-TAG_DIM], self.command()[ids]], -1))


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
      log = {**_league_log(self.league.stats(it)), **{f"league/envs/{k}": v / N for k, v in envs.items()}}
    self.steps += 1
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
      log.update({f"league/win_rate/{k}": torch.cat(v) for k, v in wins.items()})
    if n:
      robot, other = self.seats(env)
      rew = torch.cat([robot, other[:n]])
      terminated, truncated = (torch.cat([x, x[:n]]) for x in (terminated, truncated))
      log["Seat/robot_step_reward"], log["Seat/opponent_step_reward"] = robot[:n].mean(), other[:n].mean()
    extras["log"].update(log)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return self._learner(obs), rew, (terminated | truncated).to(dtype=torch.long), extras
