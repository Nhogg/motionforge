"""Two-G1 tag, derived from mjlab's stock G1 flat velocity task.

The learner keeps the entity name ``robot`` so every stock obs/reward/sensor term
applies unchanged; a second G1, ``opponent``, is driven from the host. The learner's
role (+1 pursuer, -1 evader) is a per-env command resampled at reset.
"""

import inspect
import math
from dataclasses import dataclass, replace

import torch
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz, yaw_quat

ACTION = {"robot": "joint_pos", "opponent": "opponent"}  # Entity -> action term.
TAG_DIM = 6  # rel pos xy, rel vel xy, role, distance to arena wall.
# Two default-pose G1s at base distance >= this don't touch (rules_debug/probe.py: facing
# each other clear from 0.50 m, side by side from 0.55 m).
G1_CLEARANCE = 0.6


# Hydra structured configs (groups rules/, reset/, shaping/ in config/). TagRules and
# ShapingCfg default to the legacy game; ResetCfg to the new spawn (LEGACY_RESET: the old one).
@dataclass(frozen=True)
class TagRules:
  """Game rules."""

  arena_half_extent: float = 4.0  # Square arena half-extent (m).
  episode_s: float = 20.0
  tag_mode: str = "distance"  # distance | contact.
  tag_radius: float = 0.5  # distance: base xy distance below this tags.
  tag_bodies: tuple[str, ...] = ("pelvis",)  # contact: pursuer subtrees (regex) that tag.
  win_reward: float = 10.0
  fall_loses: bool = True  # Apply own failure penalty; never award an opponent win.
  out_loses: bool = True  # Apply own failure penalty; never award an opponent win.
  learner_role: str = "random"  # random (50/50 per episode) | pursuer | evader.
  post_tag_s: float = 0.0  # Training recovery after the one-time TAG payoff; 0 = immediate reset.

  def __post_init__(self):
    assert self.tag_mode in ("distance", "contact"), self.tag_mode
    assert self.learner_role in ("random", "pursuer", "evader"), self.learner_role
    assert self.win_reward > 0 and self.episode_s > 0
    assert math.isfinite(self.post_tag_s) and self.post_tag_s >= 0
    object.__setattr__(self, "tag_bodies", tuple(self.tag_bodies))


@dataclass(frozen=True)
class ResetCfg:
  """Spawn distribution: separation d, uniform direction, pair placement, headings."""

  placement: str = "uniform"  # uniform: anywhere, both >= wall_margin inside | centered.
  separation: str = "linear"  # linear | exponential | fixed (always d_min).
  d_min: float = 0.75
  d_max: float = 5.0
  p_min_ratio: float = 0.1  # linear: density at d_max relative to d_min.
  scale: float = 1.5  # exponential: decay length (m), truncated to [d_min, d_max].
  wall_margin: float = 0.5
  heading: str = "facing"  # facing | random.
  yaw_noise: float = 0.0  # facing: uniform +-yaw_noise (rad) per robot.

  def __post_init__(self):
    assert self.placement in ("uniform", "centered"), self.placement
    assert self.separation in ("linear", "exponential", "fixed"), self.separation
    assert self.heading in ("facing", "random"), self.heading
    assert 0 < self.p_min_ratio <= 1 and self.d_min <= self.d_max


LEGACY_RESET = ResetCfg(placement="centered", separation="fixed", d_min=2.0, d_max=2.0)


@dataclass(frozen=True)
class ShapingCfg:
  """Potential-based shaping F = gamma*phi(s') - phi(s), phi(s') = 0 at terminal states."""

  enabled: bool = False
  weight: float = 1.0
  gamma: float = 0.99  # Must equal the PPO discount.
  potential: str = "distance"  # Key of POTENTIALS.

  def __post_init__(self):
    assert self.potential in POTENTIALS, self.potential
    assert math.isfinite(self.weight) and self.weight >= 0
    assert 0 < self.gamma <= 1


@dataclass(kw_only=True)
class RoleCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (1e9, 1e9)  # Only at reset.
  rules: TagRules = TagRules()  # The env's rules live here; read with tag_rules(env).
  # Render only: (pursuer, evader) RGBA tinting each robot's "silver" material per env.
  colors: tuple[tuple[float, ...], tuple[float, ...]] | None = None

  def build(self, env: ManagerBasedRlEnv) -> "RoleCommand":
    return RoleCommand(self, env)


class RoleCommand(CommandTerm):
  """Learner role per env: +1 pursuer, -1 evader, 50/50 at reset unless rules force it."""

  def __init__(self, cfg: RoleCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.role = torch.ones(self.num_envs, 1, device=self.device)
    self.tag_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
    self.tag_pulse = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.game_result = torch.zeros(self.num_envs, device=self.device)
    if cfg.colors:  # Per-world mat_rgba (the Warp renderer reads it per world).
      env.sim.expand_model_fields(("mat_rgba",))
      self.mat = [env.sim.mj_model.material(f"{k}/silver").id for k in ("robot", "opponent")]
      self.rgba = torch.tensor(cfg.colors, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.role

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.tag_step[env_ids] = -1
    # Keep game_result until the next termination computation: the league reads
    # the completed outcome after mjlab has already auto-reset these worlds.
    forced = {"pursuer": 1.0, "evader": -1.0}.get(self.cfg.rules.learner_role)
    self.role[env_ids] = torch.randint_like(self.role[env_ids], 2) * 2 - 1 if forced is None else forced
    if self.cfg.colors:
      pursuer = self.role[env_ids, 0] > 0
      for mat, chaser in zip(self.mat, (pursuer, ~pursuer)):
        self._env.sim.model.mat_rgba[env_ids, mat] = torch.where(chaser[:, None], self.rgba[0], self.rgba[1])

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


def role(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_command("role")[:, 0]


def potential(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_command("potential")[:, 0]


def tag_rules(env: ManagerBasedRlEnv) -> TagRules:
  return env.command_manager.get_term("role").cfg.rules


def distance(env: ManagerBasedRlEnv) -> torch.Tensor:
  a, b = env.scene["robot"].data, env.scene["opponent"].data
  return (a.root_link_pos_w[:, :2] - b.root_link_pos_w[:, :2]).norm(dim=-1)


def wall_distance(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  xy = env.scene[name].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  return tag_rules(env).arena_half_extent - xy.abs().max(dim=-1).values


def tag_features(env: ManagerBasedRlEnv, me: str, other: str, role_sign: float):
  """Opponent pos/vel in my yaw frame, my role, my distance to the arena wall."""
  a, b = env.scene[me].data, env.scene[other].data
  q = yaw_quat(a.root_link_quat_w)
  rel_p = quat_apply_inverse(q, b.root_link_pos_w - a.root_link_pos_w)[:, :2]
  rel_v = quat_apply_inverse(q, b.root_link_lin_vel_w - a.root_link_lin_vel_w)[:, :2]
  r = role_sign * role(env)
  return torch.cat([rel_p, rel_v, r[:, None], wall_distance(env, me)[:, None]], -1)


def episode_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
  # A recovery window must not change the pre-TAG critic observation.
  horizon = math.ceil(tag_rules(env).episode_s / env.step_dt)
  return (env.episode_length_buf / horizon)[:, None].float()


def touching(env: ManagerBasedRlEnv, sensor: str) -> torch.Tensor:
  """Any contact on the sensor now, or force in any substep of the last policy step."""
  d = env.scene[sensor].data
  return (d.found > 0).any(-1) | (d.force_history.norm(dim=-1) > 0).flatten(1).any(-1)


def tagged(env: ManagerBasedRlEnv) -> torch.Tensor:
  rules = tag_rules(env)
  if rules.tag_mode == "distance":
    return distance(env) < rules.tag_radius
  return torch.where(role(env) > 0, touching(env, "tag_robot"), touching(env, "tag_opponent"))


def out_of_arena(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return wall_distance(env, asset_cfg.name) < 0.0


def failed(env: ManagerBasedRlEnv) -> torch.Tensor:
  t = env.termination_manager.get_term
  return t("fell_over") | t("opponent_fell_over") | t("robot_out") | t("opponent_out")


def game_timeout(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Healthy game deadline, allowing a late TAG its complete recovery window."""
  state = env.command_manager.get_term("role")
  return ((env.episode_length_buf * env.step_dt >= tag_rules(env).episode_s - 1e-8)
          & (state.tag_step < 0) & ~tagged(env))


def tag_termination(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Latch the first valid TAG, pay once, and end after recovery or own failures.

  Runs LAST, after all failure/timeout terms. TAG remains the game outcome even
  if a robot later falls; that robot still receives its separate failure penalty.
  """
  state, rules = env.command_manager.get_term("role"), tag_rules(env)
  state.tag_pulse[:] = (state.tag_step < 0) & tagged(env) & ~failed(env)
  state.tag_step[state.tag_pulse] = env.episode_length_buf[state.tag_pulse]
  latched = state.tag_step >= 0
  state.game_result[:] = game_outcome(latched, env.termination_manager.get_term("time_out"),
                                      failed(env) & ~latched, role(env))
  return latched & ((env.episode_length_buf - state.tag_step) * env.step_dt >= rules.post_tag_s - 1e-8)


def game_outcome(tag: torch.Tensor, timeout: torch.Tensor, failure: torch.Tensor,
                 learner_role: torch.Tensor) -> torch.Tensor:
  """Only healthy TAG/timeout are game outcomes. Physical failures are censored."""
  result = torch.where(tag, learner_role, torch.where(timeout, -learner_role, 0.0))
  return torch.where(failure, 0.0, result)


def outcome(env: ManagerBasedRlEnv, learner_role: torch.Tensor) -> torch.Tensor:
  if tag_rules(env).post_tag_s > 0:
    return env.command_manager.get_term("role").game_result
  t = env.termination_manager.get_term
  return game_outcome(t("tagged"), t("time_out"), failed(env), learner_role)


def tag_result(env: ManagerBasedRlEnv) -> torch.Tensor:
  if tag_rules(env).post_tag_s > 0:
    state = env.command_manager.get_term("role")
    # The +/-10 payoff is paid at first contact, NOT again at recovery completion.
    return game_outcome(state.tag_pulse, env.termination_manager.get_term("time_out"),
                        failed(env) & (state.tag_step < 0), role(env))
  return outcome(env, role(env))


def own_failure(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Individual penalty, mirrored per seat (NOT a zero-sum game reward)."""
  t, rules = env.termination_manager.get_term, tag_rules(env)
  fall, out = ("fell_over", "robot_out") if asset_cfg.name == "robot" else ("opponent_fell_over", "opponent_out")
  return -((t(fall) & rules.fall_loses) | (t(out) & rules.out_loses)).float()


def motion_quality(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  """Physical diagnostics only; these do not modify actions or rewards."""
  a, b = env.scene["robot"].data, env.scene["opponent"].data
  p = role(env) > 0
  q = torch.where(p[:, None], a.root_link_quat_w, b.root_link_quat_w)
  tilt = torch.acos((1 - 2 * (q[:, 1].square() + q[:, 2].square())).clamp(-1, 1))
  angular = torch.where(p[:, None], a.root_link_ang_vel_w, b.root_link_ang_vel_w).norm(dim=-1)
  delta, velocity = b.root_link_pos_w[:, :2] - a.root_link_pos_w[:, :2], b.root_link_lin_vel_w[:, :2] - a.root_link_lin_vel_w[:, :2]
  closing = -(delta * velocity).sum(-1) / delta.norm(dim=-1).clamp_min(1e-6)
  force = torch.zeros_like(closing)
  if tag_rules(env).tag_mode == "contact":
    force = torch.stack([env.scene[name].data.force_history.norm(dim=-1).flatten(1).max(-1).values
                         for name in ("tag_robot", "tag_opponent")]).max(0).values
  return {"distance_m": delta.norm(dim=-1), "closing_speed_mps": closing,
          "pursuer_root_tilt_deg": torch.rad2deg(tilt), "pursuer_angular_speed_radps": angular,
          "interagent_peak_force_N": force}


def quality_metric(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  return motion_quality(env)[name]


def ending_metric(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  t = env.termination_manager.get_term
  hit = (env.command_manager.get_term("role").tag_step >= 0) if tag_rules(env).post_tag_s > 0 else t("tagged") & ~failed(env)
  flags = {"valid_tag": hit,
           "healthy_timeout": t("time_out") & ~hit & ~failed(env), "physical_failure": failed(env),
           "post_tag_failure": hit & failed(env), "post_tag_survived": hit & t("tagged") & ~failed(env)}
  return flags[name].float()


def reward_component(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  """True episode sums, unlike stock Episode_Reward's per-max-duration rates."""
  manager = env.reward_manager
  if name not in manager.active_terms:
    return torch.zeros(env.num_envs, device=env.device)
  scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
  return manager._step_reward[:, manager.active_terms.index(name)] * scale


def qpos_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Base xy distance from the free-joint qpos. Unlike distance(), which reads kinematics
  that lag one substep at reward time, it is the same before and after the step's forward()."""
  a, b = (env.scene[k].data for k in ("robot", "opponent"))
  xy = lambda d: d.data.qpos[:, d.indexing.free_joint_q_adr[:2]]
  return (xy(a) - xy(b)).norm(dim=-1)


POTENTIALS = {"distance": lambda env: -role(env) * qpos_distance(env)}  # Pursuer: closer is better.


@dataclass(kw_only=True)
class PotentialCfg(CommandTermCfg):
  potential: str = "distance"
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  def build(self, env: ManagerBasedRlEnv) -> "Potential":
    return Potential(self, env)


class Potential(CommandTerm):
  """phi(s) of each env's current state, recorded after every step and reset (post-forward,
  after the role resample), so the next step's shaping sees the post-reset phi(s0)."""

  def __init__(self, cfg: PotentialCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.phi = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.phi

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.phi[ids] = POTENTIALS[self.cfg.potential](self._env)[ids, None]

  def _update_metrics(self) -> None:
    pass


def shaping(env: ManagerBasedRlEnv, gamma: float, name: str) -> torch.Tensor:
  """gamma*phi(s') - phi(s) with phi(s') = 0 on terminations and timeouts, so each episode's
  discounted shaping sums to -phi(s0): no bias, and nothing carries across an auto-reset."""
  live = ~env.termination_manager.dones
  if tag_rules(env).post_tag_s > 0:
    state = env.command_manager.get_term("role")
    # Close the game potential once on TAG, then no chase reward during recovery.
    active = (state.tag_step < 0) | state.tag_pulse
    live = live & (state.tag_step < 0)
    return torch.where(active, gamma * POTENTIALS[name](env) * live - potential(env), 0.0)
  return gamma * POTENTIALS[name](env) * live - potential(env)


def action_rate_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Stock action_rate_l2 restricted to one action term (stock sums every term)."""
  am = env.action_manager
  i = am.active_terms.index(action_name)
  s = sum(am.action_term_dim[:i])
  d = (am.action - am.prev_action)[:, s : s + am.action_term_dim[i]]
  return torch.sum(torch.square(d), dim=1)


def sample_separation(cfg: ResetCfg, n: int, device) -> torch.Tensor:
  """d on [d_min, d_max] by inverse CDF; densities fall from d_min to d_max."""
  if cfg.separation == "fixed":
    return torch.full((n,), cfg.d_min, device=device)
  u, span = torch.rand(n, device=device), cfg.d_max - cfg.d_min
  if cfg.separation == "exponential":
    return cfg.d_min - cfg.scale * torch.log1p(-u * (1 - math.exp(-span / cfg.scale)))
  k = 1 - cfg.p_min_ratio  # Linear density 1 - k*x on x = (d - d_min) / span.
  x = u if k == 0 else (1 - torch.sqrt(1 - k * (2 - k) * u)) / k
  return cfg.d_min + span * x


def reset_tag_spawn(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, cfg: ResetCfg):
  """Robots d apart along a uniform direction theta, robot -> opponent, placed uniformly so
  both are >= wall_margin inside (or centered on the env origin). LEGACY_RESET draws only
  theta, reproducing the legacy spawn exactly."""
  env_ids = resolve_env_ids(env, env_ids)
  n, dev = len(env_ids), env.device
  theta = (torch.rand(n, device=dev) * 2 - 1) * math.pi
  d = sample_separation(cfg, n, dev)
  room = tag_rules(env).arena_half_extent - cfg.wall_margin
  if cfg.placement == "uniform":  # Redraw directions whose x or y extent exceeds the arena.
    while (bad := d * torch.maximum(theta.cos().abs(), theta.sin().abs()) > 2 * room).any():
      theta[bad] = (torch.rand(int(bad.sum()), device=dev) * 2 - 1) * math.pi
  half = 0.5 * d[:, None] * torch.stack([theta.cos(), theta.sin(), 0 * theta], -1)
  origin = env.scene.env_origins[env_ids]
  if cfg.placement == "uniform":
    origin = origin.clone()
    origin[:, :2] += (torch.rand(n, 2, device=dev) * 2 - 1) * (room - half[:, :2].abs())
  zero = torch.zeros_like(theta)
  yaws = [theta, theta + math.pi]
  if cfg.heading == "random":
    yaws = [(torch.rand(n, device=dev) * 2 - 1) * math.pi for _ in yaws]
  elif cfg.yaw_noise > 0:
    yaws = [y + (torch.rand(n, device=dev) * 2 - 1) * cfg.yaw_noise for y in yaws]
  for name, sign, yaw in (("robot", -1, yaws[0]), ("opponent", 1, yaws[1])):
    ent = env.scene[name]
    state = ent.data.default_root_state[env_ids].clone()
    state[:, :3] += origin + sign * half
    state[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)
    ent.write_root_state_to_sim(state, env_ids=env_ids)


def _retarget(term: ObservationTermCfg) -> ObservationTermCfg:
  """Point a learner obs term at the opponent entity."""
  p = dict(term.params)
  if "sensor_name" in p:
    p["sensor_name"] = p["sensor_name"].replace("robot/", "opponent/")
  if "action_name" in p:
    p["action_name"] = ACTION["opponent"]
  if term.func is tag_features:
    p.update(me="opponent", other="robot", role_sign=-1.0)
  elif "asset_cfg" in inspect.signature(term.func).parameters:
    p["asset_cfg"] = replace(p.get("asset_cfg", SceneEntityCfg("robot")), name="opponent")
  return replace(term, params=p)


def tag_env_cfg(
  rules: TagRules = TagRules(), reset: ResetCfg = ResetCfg(), shaping_cfg: ShapingCfg = ShapingCfg()
) -> ManagerBasedRlEnvCfg:
  """Tag task. Reward weights are per episode (divided by dt, as mjlab scales by dt)."""
  room = rules.arena_half_extent - reset.wall_margin
  assert reset.d_min > rules.tag_radius, "spawn inside the tag radius"
  assert rules.tag_mode != "contact" or reset.d_min >= G1_CLEARANCE, "spawn bodies may touch"
  assert reset.d_max <= (2 * room if reset.placement == "centered" else 2 * math.sqrt(2) * room)
  cfg = unitree_g1_flat_env_cfg()
  dt = cfg.sim.mujoco.timestep * cfg.decimation
  cfg.scene.entities["opponent"] = get_g1_robot_cfg()
  cfg.sim.njmax = 600  # Two robots.
  cfg.episode_length_s = rules.episode_s + rules.post_tag_s
  cfg.is_finite_horizon = True  # Timeout is an evader win, not a truncation.
  cfg.commands = {"role": RoleCommandCfg(rules=rules)}
  cfg.curriculum = {}
  cfg.metrics = {name: MetricsTermCfg(func=ending_metric, params={"name": name}, reduce="last")
                 for name in ("valid_tag", "healthy_timeout", "physical_failure")}
  if rules.post_tag_s > 0:
    cfg.metrics.update({name: MetricsTermCfg(func=ending_metric, params={"name": name}, reduce="last")
                        for name in ("post_tag_failure", "post_tag_survived")})
  cfg.metrics.update({name: MetricsTermCfg(func=quality_metric, params={"name": name}, reduce="last")
                      for name in ("distance_m", "closing_speed_mps", "pursuer_root_tilt_deg",
                                   "pursuer_angular_speed_radps", "interagent_peak_force_N")})

  cfg.actions["opponent"] = replace(cfg.actions["joint_pos"], entity_name="opponent")

  cfg.events.pop("push_robot")
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_tag_spawn, mode="reset", params={"cfg": reset}
  )
  joints = cfg.events["reset_robot_joints"]
  cfg.events["reset_opponent_joints"] = replace(
    joints, params={**joints.params, "asset_cfg": SceneEntityCfg("opponent", joint_names=(".*",))}
  )

  tag = ObservationTermCfg(
    func=tag_features, params={"me": "robot", "other": "opponent", "role_sign": 1.0}
  )
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    del terms["command"]
    terms["actions"] = replace(terms["actions"], params={"action_name": ACTION["robot"]})
    terms["tag"] = tag
  cfg.observations["critic"].terms["progress"] = ObservationTermCfg(func=episode_progress)
  actor = cfg.observations["actor"]
  cfg.observations["opponent"] = replace(
    actor, terms={k: _retarget(t) for k, t in actor.terms.items()}
  )

  # Command-gated regularizers: drop the ones that require a command, ungate the rest.
  for k in ("track_linear_velocity", "track_angular_velocity", "pose", "foot_swing_height", "foot_slip", "air_time"):
    cfg.rewards.pop(k)
  for k in ("foot_clearance", "soft_landing"):
    cfg.rewards[k].params["command_name"] = None
  cfg.rewards["action_rate_l2"] = replace(
    cfg.rewards["action_rate_l2"], func=action_rate_l2, params={"action_name": ACTION["robot"]}
  )
  cfg.rewards["tag"] = RewardTermCfg(func=tag_result, weight=rules.win_reward / dt)
  cfg.rewards["failure"] = RewardTermCfg(func=own_failure, weight=rules.win_reward / dt,
                                       params={"asset_cfg": SceneEntityCfg("robot")})
  if shaping_cfg.enabled:
    cfg.commands["potential"] = PotentialCfg(potential=shaping_cfg.potential)  # After role.
    cfg.rewards["shaping"] = RewardTermCfg(
      func=shaping, weight=shaping_cfg.weight / dt,
      params={"gamma": shaping_cfg.gamma, "name": shaping_cfg.potential},
    )
  cfg.metrics.update({f"reward_{name}_episode_sum": MetricsTermCfg(
    func=reward_component, params={"name": name}, reduce="sum") for name in ("tag", "shaping", "failure")})
  if rules.tag_mode == "contact":  # Pursuer's tag_bodies subtrees vs. the other's whole body.
    cfg.scene.sensors += tuple(
      ContactSensorCfg(
        name=f"tag_{me}",
        primary=ContactMatch(mode="subtree", pattern=rules.tag_bodies, entity=me),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity=other),
        fields=("found", "force"),
        reduce="none",
        history_length=cfg.decimation,  # Catch touches that resolve within a policy step.
      )
      for me, other in (("robot", "opponent"), ("opponent", "robot"))
    )

  fell = cfg.terminations["fell_over"]
  cfg.terminations.update(
    opponent_fell_over=replace(fell, params={**fell.params, "asset_cfg": SceneEntityCfg("opponent")}),
    tagged=TerminationTermCfg(func=tagged),
    robot_out=TerminationTermCfg(func=out_of_arena, params={"asset_cfg": SceneEntityCfg("robot")}),
    opponent_out=TerminationTermCfg(func=out_of_arena, params={"asset_cfg": SceneEntityCfg("opponent")}),
  )
  if rules.post_tag_s > 0:
    cfg.terminations["time_out"] = replace(cfg.terminations["time_out"], func=game_timeout, params={})
    # All current-step failures must be available before deciding valid contact.
    cfg.terminations.pop("tagged")
    cfg.terminations["tagged"] = TerminationTermCfg(func=tag_termination)
  return cfg
