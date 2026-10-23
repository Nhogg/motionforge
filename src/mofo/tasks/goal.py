"""Single-G1 goal reaching: the dense-reward, no-opponent baseline for the tag task.

Same robot, arena, termination penalties and joint-limit handling as tag (TagRules); the
opponent is replaced by a goal point, drawn at the tag spawn separation (ResetCfg) from the
robot and redrawn when reached or after ``GoalCfg.timeout_s``. The 6-D ``goal`` observation
keeps the tag layout with a stationary target: goal position and velocity relative to the
robot in its yaw frame, role (+1, a pursuer), and distance to the arena wall. Reward: metres
closed toward the goal each step.
"""

import math
from dataclasses import dataclass, replace

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz, yaw_quat

from mofo.tasks.tag import (
  GAIT_TERMS, NATURALNESS, ResetCfg, RoleCommandCfg, TagRules, alive, episode_progress, fell_over,
  limit_targets, out_of_arena, role, sample_separation, seat_end, wall_distance,
)


@dataclass(frozen=True)
class GoalCfg:
  """Goal behaviour and the dense reward (Hydra group goal/)."""

  radius: float = 0.5  # Reached when the base xy is within this (m); a new goal is drawn.
  timeout_s: float = 8.0  # A goal not reached within this is redrawn.
  progress_weight: float = 1.0  # Reward per metre closed toward the goal.


def base_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Base xy from the free-joint qpos: current right after a reset write, before forward()."""
  d = env.scene["robot"].data
  return d.data.qpos[:, d.indexing.free_joint_q_adr[:2]]


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
  goal: GoalCfg = GoalCfg()
  reset: ResetCfg = ResetCfg()
  resampling_time_range: tuple[float, float] = (8.0, 8.0)  # Set to goal.timeout_s.

  def build(self, env: ManagerBasedRlEnv) -> "GoalCommand":
    return GoalCommand(self, env)


class GoalCommand(CommandTerm):
  """World xy goal per env, and the distance to it after the last step (``prev_d``)."""

  def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.goal = torch.zeros(self.num_envs, 2, device=self.device)
    self.prev_d = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.goal

  def distance(self) -> torch.Tensor:
    return (self.goal - base_xy(self._env)).norm(dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """d from the tag spawn separation along a uniform direction; redraw directions that leave
    the arena (minus wall_margin), then fall back to a uniform point."""
    env, n, dev = self._env, len(env_ids), self.device
    origin = env.scene.env_origins[env_ids, :2]
    room = env.command_manager.get_term("role").cfg.rules.arena_half_extent - self.cfg.reset.wall_margin
    start = base_xy(env)[env_ids]
    d = sample_separation(self.cfg.reset, n, dev)
    goal = torch.empty(n, 2, device=dev)
    bad = torch.ones(n, dtype=torch.bool, device=dev)
    for _ in range(50):
      theta = (torch.rand(int(bad.sum()), device=dev) * 2 - 1) * math.pi
      goal[bad] = start[bad] + d[bad, None] * torch.stack([theta.cos(), theta.sin()], -1)
      bad = ((goal - origin).abs() > room).any(-1)
      if not bad.any():
        break
    goal[bad] = origin[bad] + (torch.rand(int(bad.sum()), 2, device=dev) * 2 - 1) * room
    self.goal[env_ids] = goal
    self.prev_d[env_ids] = (goal - start).norm(dim=-1)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    """After the step's reward: redraw reached goals, then record the distance to the goal."""
    if env_ids is None:
      self._resample((self.distance() < self.cfg.goal.radius).nonzero().flatten())
    ids = slice(None) if env_ids is None else env_ids
    self.prev_d[ids] = self.distance()[ids]

  def _update_metrics(self) -> None:
    pass


def goal_term(env: ManagerBasedRlEnv) -> GoalCommand:
  return env.command_manager.get_term("goal")


def goal_features(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Tag's 6-D layout with a stationary target: goal pos, goal-minus-own velocity (yaw frame),
  role, wall distance."""
  d = env.scene["robot"].data
  q = yaw_quat(d.root_link_quat_w)
  rel = torch.cat([goal_term(env).goal - base_xy(env), torch.zeros_like(d.root_link_pos_w[:, :1])], -1)
  rel_p = quat_apply_inverse(q, rel)[:, :2]
  rel_v = quat_apply_inverse(q, -d.root_link_lin_vel_w)[:, :2]
  return torch.cat([rel_p, rel_v, role(env)[:, None], wall_distance(env, "robot")[:, None]], -1)


def goal_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Speed toward the goal over the step (m/s); x weight x dt = metres closed."""
  g = goal_term(env)
  return (g.prev_d - g.distance()) / env.step_dt


def goal_reached(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 on the step a goal is reached (it is redrawn right after); summed per episode."""
  g = goal_term(env)
  return (g.distance() < g.cfg.goal.radius).float()


def reset_goal_spawn(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, cfg: ResetCfg):
  """Robot anywhere >= wall_margin inside the arena, uniform heading."""
  env_ids = resolve_env_ids(env, env_ids)
  n, dev = len(env_ids), env.device
  room = env.command_manager.get_term("role").cfg.rules.arena_half_extent - cfg.wall_margin
  ent = env.scene["robot"]
  state = ent.data.default_root_state[env_ids].clone()
  state[:, :3] += env.scene.env_origins[env_ids]
  state[:, :2] += (torch.rand(n, 2, device=dev) * 2 - 1) * room
  yaw = (torch.rand(n, device=dev) * 2 - 1) * math.pi
  state[:, 3:7] = quat_from_euler_xyz(0 * yaw, 0 * yaw, yaw)
  ent.write_root_state_to_sim(state, env_ids=env_ids)


def goal_env_cfg(rules: TagRules = TagRules(), reset: ResetCfg = ResetCfg(), goal: GoalCfg = GoalCfg()) -> ManagerBasedRlEnvCfg:
  """Goal task with tag's penalties. Reward weights are per episode (divided by dt, as in tag)."""
  cfg = unitree_g1_flat_env_cfg()
  dt = cfg.sim.mujoco.timestep * cfg.decimation
  if rules.limit_targets:
    cfg.scene.entities["robot"] = limit_targets(cfg.scene.entities["robot"])
  cfg.episode_length_s = rules.episode_s  # Timeout is a truncation here (bootstrapped).
  # "role" holds the rules (wall distance, arena exits) and a constant +1 role input.
  cfg.commands = {
    "role": RoleCommandCfg(rules=replace(rules, learner_role="pursuer")),
    "goal": GoalCommandCfg(goal=goal, reset=reset, resampling_time_range=(goal.timeout_s, goal.timeout_s)),
  }
  cfg.curriculum = {}
  cfg.metrics = {k: MetricsTermCfg(func=f, params=p, reduce=r) for k, (f, p, r) in NATURALNESS.items()}
  cfg.metrics["goals_reached"] = MetricsTermCfg(func=goal_reached, reduce="sum")

  cfg.events.pop("push_robot")
  cfg.events["reset_base"] = EventTermCfg(func=reset_goal_spawn, mode="reset", params={"cfg": reset})

  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    del terms["command"]
    terms["goal"] = ObservationTermCfg(func=goal_features)
  cfg.observations["critic"].terms["progress"] = ObservationTermCfg(func=episode_progress)

  # As tag: command-gated terms out, and with regularizers off only dof_pos_limits (if weighted) stays.
  for k in ("track_linear_velocity", "track_angular_velocity", "pose", "foot_swing_height", "foot_slip", "air_time"):
    cfg.rewards.pop(k)
  for k in ("foot_clearance", "soft_landing"):
    cfg.rewards[k].params["command_name"] = None
  if rules.joint_limit_weight is not None:
    cfg.rewards["dof_pos_limits"] = replace(cfg.rewards["dof_pos_limits"], weight=rules.joint_limit_weight)
  if not rules.regularizers:
    for k in set(cfg.rewards) - set(GAIT_TERMS) - ({"dof_pos_limits"} if rules.joint_limit_weight is not None else set()):
      cfg.rewards.pop(k)
  cfg.rewards["progress"] = RewardTermCfg(func=goal_progress, weight=goal.progress_weight)
  for kind, total in (("fell", rules.fall_reward), ("out", rules.out_reward)):
    if total is not None:  # No game outcome to add to: the total is the penalty.
      cfg.rewards[f"{kind}_extra"] = RewardTermCfg(func=seat_end, weight=total / dt, params={"kind": kind})
  if rules.alive_reward:
    cfg.rewards["alive"] = RewardTermCfg(func=alive, weight=rules.alive_reward / dt)

  fell = cfg.terminations["fell_over"]
  if rules.fall_height is not None:
    cfg.terminations["fell_over"] = TerminationTermCfg(
      func=fell_over, params={**fell.params, "min_height": rules.fall_height, "asset_cfg": SceneEntityCfg("robot")}
    )
  cfg.terminations.update(
    robot_out=TerminationTermCfg(func=out_of_arena, params={"asset_cfg": SceneEntityCfg("robot")}),
    nan_reset=TerminationTermCfg(func=nan_detection),
  )
  return cfg
