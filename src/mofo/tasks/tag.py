"""Two-G1 tag, derived from mjlab's stock G1 flat velocity task.

The learner keeps the entity name ``robot`` so every stock obs/reward/sensor term
applies unchanged; a second G1, ``opponent``, is driven from the host. The learner's
role (+1 pursuer, -1 evader) is a per-env command resampled at reset.
"""

import inspect
import math
from dataclasses import dataclass, field, fields, replace

import mujoco
import torch
from mjlab.actuator import BuiltinPositionActuator, BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_group import _TARGET_TENSOR_MAP
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.envs.mdp.terminations import bad_orientation, nan_detection
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
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
  fall_loses: bool = True  # Otherwise falling ends the episode as a draw.
  out_loses: bool = True  # Otherwise leaving the arena ends the episode as a draw.
  learner_role: str = "random"  # random (50/50 per episode) | pursuer | evader.
  fall_reward: float | None = None  # A seat's reward when it falls (null: -win_reward).
  out_reward: float | None = None  # A seat's reward when it leaves the arena (null: -win_reward).
  tag_confirm_s: float = 0.0  # A tag counts once the pursuer stays up this long after contact.
  alive_reward: float = 0.0  # Per-step reward per seat while upright.
  clock_reward: float = 0.0  # Zero-sum, per second of game: the evader gets +clock_reward, the pursuer -clock_reward.
  timeout_wins: bool = True  # The timeout is an evader win; false: a draw (no reward).
  regularizers: bool = True  # Stock mjlab regularizers (upright, action rate, ...); false: none.
  reward_weights: dict[str, float] = field(default_factory=dict)  # Per-term weight overrides (stock regularizers); 0 drops the term.
  joint_limit_weight: float | None = None  # Stock dof_pos_limits weight (null: stock); kept even without regularizers.
  limit_targets: bool = False  # Clamp actuator position targets to the joint ranges (ctrllimited).
  fall_height: float | None = None  # A base (pelvis) at or below this height (m) also counts as a fall.
  stance_slip_weight: float = 0.0  # Per second x mean over the feet of contact x |foot v_xy|^2 (stance feet only).
  tilt_weight: float = 0.0  # Per second x max(torso tilt - tilt_free, 0)^2 (rad^2).
  tilt_free: float = 0.35  # Tilt (rad) that costs nothing, room for a running lean.
  decimation: int | None = None  # Physics steps per policy step (null: stock 4, i.e. 50 Hz at the 5 ms timestep).

  def __post_init__(self):
    assert self.tag_mode in ("distance", "contact"), self.tag_mode
    assert self.learner_role in ("random", "pursuer", "evader"), self.learner_role
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
  gamma: float = 0.99  # Must equal the PPO discount when pbrs.
  potential: str = "distance"  # Key of POTENTIALS.
  pbrs: bool = True  # False: phi(s') is not zeroed at terminal states, so F is the plain
  # per-step change in phi (biased: an episode sums to phi(s_T) - phi(s0) at gamma=1).

  def __post_init__(self):
    assert self.potential in POTENTIALS, self.potential


@dataclass
class GaitWeights:  # Not frozen: a frozen nested node would be read-only to Hydra overrides.
  """Per-term weight overrides; null: the stock stage-1 weight x GaitCfg.scale."""

  pose: float | None = None
  foot_slip: float | None = None
  foot_swing_height: float | None = None
  air_time: float | None = None  # Stock weight 0 (off), so it needs an explicit weight here.


@dataclass(frozen=True)
class GaitCfg:
  """Stock stage-1 gait terms (pose, foot_slip, foot_swing_height) on a stand-in command: the
  robot's own measured velocity, i.e. "move well at whatever velocity you're moving"."""

  enabled: bool = False
  scale: float = 0.1  # x the stock unitree_g1_flat_env_cfg weights.
  weights: GaitWeights = field(default_factory=GaitWeights)


GAIT_TERMS = ("pose", "foot_slip", "foot_swing_height", "air_time")


@dataclass(kw_only=True)
class MeasuredVelocityCfg(CommandTermCfg):
  entity: str = "robot"
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  def build(self, env: ManagerBasedRlEnv) -> "MeasuredVelocity":
    return MeasuredVelocity(self, env)


class MeasuredVelocity(CommandTerm):
  """Stand-in velocity command (lin x, lin y in the yaw frame, ang z), copied from the entity's
  base after every step and reset. Not an observation: the stock gait terms read it by name."""

  def __init__(self, cfg: MeasuredVelocityCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.vel = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    d = self._env.scene[self.cfg.entity].data
    v = torch.cat([quat_apply_inverse(yaw_quat(d.root_link_quat_w), d.root_link_lin_vel_w)[:, :2], d.root_link_ang_vel_w[:, 2:]], -1)
    ids = slice(None) if env_ids is None else env_ids
    self.vel[ids] = v[ids]

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class RoleCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (1e9, 1e9)  # Only at reset.
  rules: TagRules = TagRules()  # The env's rules live here; read with tag_rules(env).
  # Render only: (pursuer, evader) RGBA tinting each robot's "silver" material per env.
  colors: tuple[tuple[float, ...], tuple[float, ...]] | None = None

  def build(self, env: ManagerBasedRlEnv) -> "RoleCommand":
    return RoleCommand(self, env)


class RoleCommand(CommandTerm):
  """Learner role per env: +1 pursuer, -1 evader, drawn at reset (evader with probability
  ``p_evader``, 0.5 unless the trainer adapts it) unless rules force it."""

  def __init__(self, cfg: RoleCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.role = torch.ones(self.num_envs, 1, device=self.device)
    self.p_evader = 0.5
    if cfg.colors:  # Per-world mat_rgba (the Warp renderer reads it per world).
      env.sim.expand_model_fields(("mat_rgba",))
      self.mat = [env.sim.mj_model.material(f"{k}/silver").id for k in ("robot", "opponent")]
      self.rgba = torch.tensor(cfg.colors, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.role

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    forced = {"pursuer": 1.0, "evader": -1.0}.get(self.cfg.rules.learner_role)
    if forced is None:
      self.role[env_ids] = torch.where(torch.rand_like(self.role[env_ids]) < self.p_evader, -1.0, 1.0)
    else:
      self.role[env_ids] = forced
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
  return (env.episode_length_buf / env.max_episode_length)[:, None].float()


def touching(env: ManagerBasedRlEnv, sensor: str) -> torch.Tensor:
  """Any contact on the sensor now, or force in any substep of the last policy step."""
  d = env.scene[sensor].data
  return (d.found > 0).any(-1) | (d.force_history.norm(dim=-1) > 0).flatten(1).any(-1)


def tagged(env: ManagerBasedRlEnv) -> torch.Tensor:
  rules = tag_rules(env)
  if rules.tag_mode == "distance":
    return distance(env) < rules.tag_radius
  return torch.where(role(env) > 0, touching(env, "tag_robot"), touching(env, "tag_opponent"))


class tag_confirmed:
  """Tag termination when ``tag_confirm_s`` > 0: the first contact locks the tag and starts a
  window; the episode runs on and the tag ends it once the pursuer has stayed up for
  ``tag_confirm_s``, or at the timeout (a pending tag counts if the pursuer is still up). A
  fall/exit inside the window ends it at once through the normal fall/out terminations, which
  outcome() ranks above the tag: the pursuer's loses, the evader's still loses to the tag.
  outcome() and Phi(terminal)=0 need no change."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self.start = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    self.steps = round(tag_rules(env).tag_confirm_s / env.step_dt)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    t = env.episode_length_buf
    self.start = torch.where(tagged(env) & (self.start < 0), t, self.start)
    return (self.start >= 0) & ((t - self.start >= self.steps) | (t >= env.max_episode_length))

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.start[slice(None) if env_ids is None else env_ids] = -1


def _pending_down(env: ManagerBasedRlEnv, seat_sign: float) -> torch.Tensor:
  """A locked tag is pending and the pursuer (+1) / evader (-1) fell or left this step."""
  t, start = env.termination_manager.get_term, env.termination_manager.get_term_cfg("tagged").func.start
  robot = t("fell_over") | t("robot_out")
  other = t("opponent_fell_over") | t("opponent_out")
  return (start >= 0) & torch.where(seat_sign * role(env) > 0, robot, other)


def tag_confirmed_now(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 where a tag won: confirmed by the window, or standing because the evader went down in it."""
  return (env.termination_manager.get_term("tagged") | (_pending_down(env, -1.0) & ~_pending_down(env, 1.0))).float()


def tag_confirm_fail(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 where a locked tag ended because the pursuer fell or left inside the window."""
  return _pending_down(env, 1.0).float()


def pose_deviation(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Mean |q - q_default| over the selected joints."""
  return mdp.joint_pos_rel(env, asset_cfg=asset_cfg).abs().mean(-1)


def foot_slip_speed(env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Mean foot xy speed over the feet in contact (0 if none)."""
  c = (env.scene[sensor_name].data.found > 0).float()
  v = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2].norm(dim=-1)
  return (v * c).sum(-1) / c.sum(-1).clamp_min(1)


def stance_slip(env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Mean over the feet of contact x |foot xy velocity|^2: swing feet cost nothing. Not gated on a command."""
  c = (env.scene[sensor_name].data.found > 0).float()
  v = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  return (c * v.square().sum(-1)).mean(-1)


def tilt_excess(env: ManagerBasedRlEnv, free: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """max(tilt - free, 0)^2, tilt from the base's projected gravity (rad)."""
  g = env.scene[asset_cfg.name].data.projected_gravity_b
  return (torch.acos((-g[:, 2]).clamp(-1, 1)) - free).clamp_min(0).square()


def base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def base_drop(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How far the base is below its default height (episode max: the lowest point)."""
  return env.scene["robot"].data.default_root_state[:, 2] - base_height(env)


def torso_tilt(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Base tilt from vertical (rad)."""
  return torch.acos((-mdp.projected_gravity(env)[:, 2]).clamp(-1, 1))


def feet_contact_ratio(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  return (env.scene[sensor_name].data.found > 0).float().mean(-1)


def feet_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Mean duration of each foot's last swing (s)."""
  return env.scene[sensor_name].data.last_air_time.mean(-1)


def action_magnitude(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.get_term(ACTION["robot"]).raw_action.abs().mean(-1)


FEET = {"sensor_name": "feet_ground_contact"}
NATURALNESS = {  # Name -> (func, params, episode reduce); logged as Episode_Metrics/*, and in eval JSONs.
  "pose_dev": (pose_deviation, {"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}, "mean"),
  "arm_pose_dev": (pose_deviation, {"asset_cfg": SceneEntityCfg("robot", joint_names=(".*(shoulder|elbow|wrist).*",))}, "mean"),
  "foot_slip_speed": (foot_slip_speed, {**FEET, "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))}, "mean"),
  "base_height": (base_height, {}, "mean"),
  "base_drop": (base_drop, {}, "max"),  # Episode log: the max (lowest point); eval JSON: per-step mean.
  "torso_tilt": (torso_tilt, {}, "mean"),
  "feet_contact_ratio": (feet_contact_ratio, FEET, "mean"),
  "feet_air_time": (feet_air_time, FEET, "mean"),
  "action_magnitude": (action_magnitude, {}, "mean"),
}


def seat_end(env: ManagerBasedRlEnv, kind: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """1.0 where this seat's own fall ("fell") or exit ("out") ended the episode."""
  name = {"fell": "fell_over", "out": "robot_out"}[kind]
  return env.termination_manager.get_term(name if asset_cfg.name == "robot" else f"opponent_{name.removeprefix('robot_')}").float()


def clock(env: ManagerBasedRlEnv) -> torch.Tensor:
  """+1 for the evader seat, -1 for the pursuer (robot seat; zero-sum, so negated for the opponent)."""
  return -role(env)


def alive(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  return 1.0 - seat_end(env, "fell", asset_cfg)


def fell_over(env: ManagerBasedRlEnv, limit_angle: float, min_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """The stock tilt check (bad_orientation), or the base at or below ``min_height``."""
  z = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return bad_orientation(env, limit_angle, asset_cfg) | (z <= min_height)


def out_of_arena(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return wall_distance(env, asset_cfg.name) < 0.0


def outcome(env: ManagerBasedRlEnv, learner_role: torch.Tensor) -> torch.Tensor:
  """+1 learner win, -1 loss, 0 if the episode goes on. Read after terminations."""
  t, rules = env.termination_manager.get_term, tag_rules(env)
  lose = (t("fell_over") & rules.fall_loses) | (t("robot_out") & rules.out_loses)
  win = (t("opponent_fell_over") & rules.fall_loses) | (t("opponent_out") & rules.out_loses)
  r = learner_role
  end = torch.where(t("tagged"), r, torch.where(t("time_out") & rules.timeout_wins, -r, torch.zeros_like(r)))
  return torch.where(lose, -1.0, torch.where(win, 1.0, end))


def tag_result(env: ManagerBasedRlEnv) -> torch.Tensor:
  return outcome(env, role(env))


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


def shaping(env: ManagerBasedRlEnv, gamma: float, name: str, pbrs: bool = True) -> torch.Tensor:
  """gamma*phi(s') - phi(s) with phi(s') = 0 on terminations and timeouts, so each episode's
  discounted shaping sums to -phi(s0): no bias, and nothing carries across an auto-reset.
  ``pbrs=False`` keeps phi(s') at terminal steps too (the plain change in phi)."""
  live = ~env.termination_manager.dones | (not pbrs)
  return torch.where(live, gamma * POTENTIALS[name](env), 0.0) - potential(env)  # NaN-safe at nan_reset.


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


@dataclass(kw_only=True)
class LimitedPositionActuatorCfg(BuiltinPositionActuatorCfg):
  def build(self, entity, target_ids: list[int], target_names: list[str]) -> "LimitedPositionActuator":
    return LimitedPositionActuator(self, entity, target_ids, target_names)


class LimitedPositionActuator(BuiltinPositionActuator):
  """MuJoCo position actuator with ctrlrange = the joint range and ctrllimited on: targets
  past a joint stop are clamped to it (mjlab leaves them free so the PD can press on the stop)."""

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    super().edit_spec(spec, target_names)
    for a, name in zip(self._mjs_actuators, target_names):
      a.ctrllimited = True
      a.ctrlrange[:] = spec.joint(name).range


# mjlab looks up a builtin actuator's target tensor by its exact type: same as the base class.
_TARGET_TENSOR_MAP[(LimitedPositionActuator, TransmissionType.JOINT)] = _TARGET_TENSOR_MAP[(BuiltinPositionActuator, TransmissionType.JOINT)]


def limit_targets(entity):
  """The entity cfg with its builtin position actuators as LimitedPositionActuatorCfg."""
  lim = lambda a: LimitedPositionActuatorCfg(**{f.name: getattr(a, f.name) for f in fields(a) if f.init})
  acts = tuple(lim(a) if type(a) is BuiltinPositionActuatorCfg else a for a in entity.articulation.actuators)
  return replace(entity, articulation=replace(entity.articulation, actuators=acts))


def tag_env_cfg(
  rules: TagRules = TagRules(), reset: ResetCfg = ResetCfg(), shaping_cfg: ShapingCfg = ShapingCfg(),
  gait: GaitCfg = GaitCfg(),
) -> ManagerBasedRlEnvCfg:
  """Tag task. Reward weights are per episode (divided by dt, as mjlab scales by dt)."""
  room = rules.arena_half_extent - reset.wall_margin
  assert reset.d_min > rules.tag_radius, "spawn inside the tag radius"
  assert rules.tag_mode != "contact" or reset.d_min >= G1_CLEARANCE, "spawn bodies may touch"
  assert reset.d_max <= (2 * room if reset.placement == "centered" else 2 * math.sqrt(2) * room)
  cfg = unitree_g1_flat_env_cfg()
  if rules.decimation is not None:  # Control rate; per-second weights below follow dt.
    cfg.decimation = rules.decimation
  dt = cfg.sim.mujoco.timestep * cfg.decimation
  cfg.scene.entities["opponent"] = get_g1_robot_cfg()
  if rules.limit_targets:
    for k in ("robot", "opponent"):
      cfg.scene.entities[k] = limit_targets(cfg.scene.entities[k])
  cfg.sim.njmax = 600  # Two robots.
  cfg.episode_length_s = rules.episode_s
  cfg.is_finite_horizon = True  # Timeout is an evader win, not a truncation.
  cfg.commands = {"role": RoleCommandCfg(rules=rules)}
  cfg.curriculum = {}
  cfg.metrics = {k: MetricsTermCfg(func=f, params=p, reduce=r) for k, (f, p, r) in NATURALNESS.items()}  # Learner seat.

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
  stock = {k: cfg.rewards.pop(k) for k in ("track_linear_velocity", "track_angular_velocity", "pose", "foot_swing_height", "foot_slip", "air_time")}
  if gait.enabled:  # Back on the stand-in command "twist" (the name the stock terms read).
    cfg.commands["twist"] = MeasuredVelocityCfg()
    for k in GAIT_TERMS:
      w = getattr(gait.weights, k)
      w = stock[k].weight * gait.scale if w is None else w
      if w:
        cfg.rewards[k] = replace(stock[k], weight=w)
  for k in ("foot_clearance", "soft_landing"):
    cfg.rewards[k].params["command_name"] = None
  cfg.rewards["action_rate_l2"] = replace(
    cfg.rewards["action_rate_l2"], func=action_rate_l2, params={"action_name": ACTION["robot"]}
  )
  if rules.joint_limit_weight is not None:
    cfg.rewards["dof_pos_limits"] = replace(cfg.rewards["dof_pos_limits"], weight=rules.joint_limit_weight)
  if not rules.regularizers:  # Game rewards only (gait terms, and dof_pos_limits if weighted, stay).
    for k in set(cfg.rewards) - set(GAIT_TERMS) - ({"dof_pos_limits"} if rules.joint_limit_weight is not None else set()):
      cfg.rewards.pop(k)
  for k, w in rules.reward_weights.items():
    assert k in cfg.rewards, f"rules.reward_weights: no reward term {k!r} (have {sorted(cfg.rewards)})"
    if w:
      cfg.rewards[k] = replace(cfg.rewards[k], weight=w)
    else:
      cfg.rewards.pop(k)
  cfg.rewards["tag"] = RewardTermCfg(func=tag_result, weight=rules.win_reward / dt)
  if rules.stance_slip_weight:  # Per second (mjlab scales by dt); mirrored onto the opponent seat in self-play.
    cfg.rewards["stance_slip"] = RewardTermCfg(func=stance_slip, weight=rules.stance_slip_weight, params={
      **FEET, "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))})
  if rules.tilt_weight:
    cfg.rewards["tilt_excess"] = RewardTermCfg(func=tilt_excess, weight=rules.tilt_weight, params={
      "free": rules.tilt_free, "asset_cfg": SceneEntityCfg("robot")})
  # Own fall/exit on top of the tag term's -win_reward (0 if it is a draw): totals fall/out_reward.
  for kind, total, loses in (("fell", rules.fall_reward, rules.fall_loses), ("out", rules.out_reward, rules.out_loses)):
    if total is not None:
      extra = total + (rules.win_reward if loses else 0.0)
      cfg.rewards[f"{kind}_extra"] = RewardTermCfg(func=seat_end, weight=extra / dt, params={"kind": kind})
  if rules.alive_reward:
    cfg.rewards["alive"] = RewardTermCfg(func=alive, weight=rules.alive_reward / dt)
  if rules.clock_reward:  # Per second: mjlab scales by dt.
    cfg.rewards["clock"] = RewardTermCfg(func=clock, weight=rules.clock_reward)
  if shaping_cfg.enabled:
    cfg.commands["potential"] = PotentialCfg(potential=shaping_cfg.potential)  # After role.
    cfg.rewards["shaping"] = RewardTermCfg(
      func=shaping, weight=shaping_cfg.weight / dt,
      params={"gamma": shaping_cfg.gamma, "name": shaping_cfg.potential, "pbrs": shaping_cfg.pbrs},
    )
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
  if rules.fall_height is not None:
    fell = cfg.terminations["fell_over"] = TerminationTermCfg(
      func=fell_over, params={**fell.params, "min_height": rules.fall_height, "asset_cfg": SceneEntityCfg("robot")}
    )
  cfg.terminations.update(
    opponent_fell_over=replace(fell, params={**fell.params, "asset_cfg": SceneEntityCfg("opponent")}),
    tagged=TerminationTermCfg(func=tag_confirmed if rules.tag_confirm_s > 0 else tagged),
    robot_out=TerminationTermCfg(func=out_of_arena, params={"asset_cfg": SceneEntityCfg("robot")}),
    opponent_out=TerminationTermCfg(func=out_of_arena, params={"asset_cfg": SceneEntityCfg("opponent")}),
    # Physics blew up (NaN/Inf qpos/qvel/qacc/sensordata): reset that env as a draw (outcome 0;
    # NaN rewards are zeroed by the reward manager). Counted as Episode_Termination/nan_reset.
    nan_reset=TerminationTermCfg(func=nan_detection),
  )
  if rules.tag_confirm_s > 0:  # Per-episode rates, logged as Episode_Metrics/*.
    cfg.metrics.update(
      tag_confirmed=MetricsTermCfg(func=tag_confirmed_now, reduce="max"),
      tag_confirm_fail=MetricsTermCfg(func=tag_confirm_fail, reduce="max"),
    )
  return cfg
