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
from mjlab.envs.mdp.rewards import posture as mdp_posture
from mjlab.utils.lab_api.string import resolve_matching_names_values
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
TAG_DIM = 6  # Legacy prefix: rel pos xy, rel vel xy, role, distance to arena wall.
# Two default-pose G1s at base distance >= this don't touch (rules_debug/probe.py: facing
# each other clear from 0.50 m, side by side from 0.55 m).
G1_CLEARANCE = 0.6


# Hydra structured configs (groups rules/, reset/, shaping/ in config/). TagRules and
# ShapingCfg default to the legacy game; ResetCfg to the new spawn (LEGACY_RESET: the old one).
@dataclass(frozen=True)
class TagRules:
  """Game rules."""

  arena_half_extent: float = 4.0  # Square arena half-extent (m).
  unbounded: bool = False  # No walls: never out of arena; the wall-distance input reads as if at the center.
  scripted_evader: str = "flee"  # flee (away from the pursuer) | random (random direction and speed, redrawn at random times).
  episode_s: float = 20.0
  tag_mode: str = "distance"  # distance | contact.
  tag_radius: float = 0.5  # distance: base xy distance below this tags.
  tag_bodies: tuple[str, ...] = ("pelvis",)  # contact: pursuer subtrees (regex) that tag.
  tag_match: str = "subtree"  # subtree: tag_bodies and their descendants | body: only those bodies' own geoms.
  win_reward: float = 10.0
  fall_loses: bool = True  # Apply own failure penalty; never award an opponent win.
  out_loses: bool = True  # Apply own failure penalty; never award an opponent win.
  pursuer_out_wins: bool = False  # A pre-TAG pursuer arena exit is an evader game win (unless the evader also failed).
  failure_loses_game: bool = False  # Any pre-TAG own fall/arena exit is a game loss (opponent win) unless both failed.
  failure_time_penalty: float = 0.0  # Extra own-failure cost per remaining game second (pre-TAG only).
  learner_role: str = "random"  # random (50/50 per episode) | pursuer | evader.
  arena_observation: bool = False # Append body-frame center xy and arena +X axis xy (four inputs).
  post_tag_s: float = 0.0  # Training recovery after the one-time TAG payoff; 0 = immediate reset.
  post_tag_failure: str = "both"  # Seats penalized for failing during recovery: both | pursuer.
  post_tag_worlds: str = "all"  # all | learner_pursuer: recover only where the learner pursues a frozen opponent.
  contact_speed_weight: float = 0.0 # Own inward-speed cost at valid first TAG (not zero-sum).
  contact_speed_limit_mps: float = 1.0
  contact_speed_max_penalty: float = 2.0
  tag_posture_weight: float = 0.0 # Pursuer's own spin/tilt cost at valid first TAG (not zero-sum).
  tag_spin_limit_radps: float = 1.5
  tag_tilt_limit_deg: float = 20.0
  tag_posture_max_penalty: float = 5.0
  upright_reward_offset: float = 0.0 # TAG-only ablation: 1 makes perfect upright worth zero, not a living bonus.
  upright_offset_roles: str = "both" # Seats charged the offset: both | pursuer (the evader keeps the living bonus).
  arm_pose_weight: float = 0.0 # Per-second cost (both seats) for arm joints far from the default pose, in [0, w].
  body_pose_weight: float = 0.0 # Same cost over all joints (locomotion running stds): crouches and lunges, not just arms.

  def __post_init__(self):
    assert self.tag_mode in ("distance", "contact"), self.tag_mode
    assert self.scripted_evader in ("flee", "random"), self.scripted_evader
    assert self.learner_role in ("random", "pursuer", "evader"), self.learner_role
    assert self.post_tag_failure in ("both", "pursuer"), self.post_tag_failure
    assert self.post_tag_worlds in ("all", "learner_pursuer"), self.post_tag_worlds
    assert self.upright_offset_roles in ("both", "pursuer"), self.upright_offset_roles
    assert self.tag_match in ("subtree", "body"), self.tag_match
    assert math.isfinite(self.arm_pose_weight) and self.arm_pose_weight >= 0
    assert math.isfinite(self.body_pose_weight) and self.body_pose_weight >= 0
    assert math.isfinite(self.failure_time_penalty) and self.failure_time_penalty >= 0
    assert self.win_reward > 0 and self.episode_s > 0
    assert math.isfinite(self.post_tag_s) and self.post_tag_s >= 0
    assert math.isfinite(self.contact_speed_weight) and self.contact_speed_weight >= 0
    assert math.isfinite(self.contact_speed_limit_mps) and self.contact_speed_limit_mps >= 0
    assert math.isfinite(self.contact_speed_max_penalty) and self.contact_speed_max_penalty > 0
    assert math.isfinite(self.upright_reward_offset) and 0 <= self.upright_reward_offset <= 1
    assert math.isfinite(self.tag_posture_weight) and self.tag_posture_weight >= 0
    assert self.tag_spin_limit_radps > 0 and self.tag_tilt_limit_deg > 0
    assert 0 < self.tag_posture_max_penalty < self.win_reward, "Posture cost must remain smaller than TAG payoff"
    if self.contact_speed_weight:
      assert self.tag_mode == "contact", self.tag_mode
      assert self.contact_speed_max_penalty < self.win_reward, "Contact cost must remain smaller than TAG payoff"
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
    self.inward_speed = torch.zeros(self.num_envs, 2, device=self.device)
    self.forced = torch.zeros(self.num_envs, device=self.device)  # Per-world role override (0: rules).
    self.frozen = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)  # Opponent seat not learning.
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
    roles = torch.randint_like(self.role[env_ids], 2) * 2 - 1 if forced is None else torch.full_like(self.role[env_ids], forced)
    world = self.forced[env_ids, None]
    self.role[env_ids] = torch.where(world != 0, world, roles)
    if self.cfg.colors:
      pursuer = self.role[env_ids, 0] > 0
      for mat, chaser in zip(self.mat, (pursuer, ~pursuer)):
        self._env.sim.model.mat_rgba[env_ids, mat] = torch.where(chaser[:, None], self.rgba[0], self.rgba[1])

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    # Called after physics/forward/reset. Cache these velocities for the NEXT
    # control step's reward, so contact impulses cannot hide pre-contact speed.
    ids = slice(None) if env_ids is None else env_ids
    for i, name in enumerate(("robot", "opponent")):
      self.inward_speed[ids, i] = own_inward_speed(self._env, name)[ids]

  def _update_metrics(self) -> None:
    pass


def role(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_command("role")[:, 0]


def potential(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_command("potential")[:, 0]


def tag_rules(env: ManagerBasedRlEnv) -> TagRules:
  return env.command_manager.get_term("role").cfg.rules


def tag_observation_dim(env: ManagerBasedRlEnv) -> int:
  return TAG_DIM + (4 if tag_rules(env).arena_observation else 0)


def distance(env: ManagerBasedRlEnv) -> torch.Tensor:
  a, b = env.scene["robot"].data, env.scene["opponent"].data
  return (a.root_link_pos_w[:, :2] - b.root_link_pos_w[:, :2]).norm(dim=-1)


def wall_distance(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  xy = env.scene[name].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  if tag_rules(env).unbounded:
    return torch.full_like(xy[:, 0], tag_rules(env).arena_half_extent)
  return tag_rules(env).arena_half_extent - xy.abs().max(dim=-1).values


def tag_features(env: ManagerBasedRlEnv, me: str, other: str, role_sign: float):
  """Legacy opponent/role/wall prefix; optional arena geometry in my yaw frame."""
  a, b = env.scene[me].data, env.scene[other].data
  q = yaw_quat(a.root_link_quat_w)
  rel_p = quat_apply_inverse(q, b.root_link_pos_w - a.root_link_pos_w)[:, :2]
  rel_v = quat_apply_inverse(q, b.root_link_lin_vel_w - a.root_link_lin_vel_w)[:, :2]
  r = role_sign * role(env)
  features = [rel_p, rel_v, r[:, None], wall_distance(env, me)[:, None]]
  if tag_rules(env).arena_observation:
    home = env.scene.env_origins - a.root_link_pos_w
    home[:, 2] = 0
    axis = torch.zeros_like(home)
    axis[:, 0] = 1
    # Center direction alone omits the square arena's orientation. Together these
    # four values locate all walls, without changing the six legacy features.
    features += [quat_apply_inverse(q, home)[:, :2] / tag_rules(env).arena_half_extent,
                 quat_apply_inverse(q, axis)[:, :2]]
  return torch.cat(features, -1)


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
                                      failed(env) & ~latched, role(env), forfeit(env) * ~latched)
  return latched & ((env.episode_length_buf - state.tag_step) * env.step_dt >= recovery_s(env) - 1e-8)


def recovery_s(env: ManagerBasedRlEnv) -> torch.Tensor | float:
  """Per-world post-TAG window. learner_pursuer: no learning evader ever sees recovery,
  so it cannot learn to fall there and carry that into pre-TAG play (no phase input)."""
  rules, state = tag_rules(env), env.command_manager.get_term("role")
  if rules.post_tag_worlds == "all":
    return rules.post_tag_s
  return torch.where(state.frozen & (role(env) > 0), rules.post_tag_s, 0.0)


def game_outcome(tag: torch.Tensor, timeout: torch.Tensor, failure: torch.Tensor,
                 learner_role: torch.Tensor, winner: torch.Tensor | None = None) -> torch.Tensor:
  """Healthy TAG/timeout are game outcomes; failures are censored unless ``winner`` (forfeit) names a role."""
  result = torch.where(tag, learner_role, torch.where(timeout, -learner_role, 0.0))
  result = torch.where(failure, 0.0, result)
  return result if winner is None else torch.where(winner != 0, winner * learner_role, result)


def forfeit(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Winning role (+1 pursuer, -1 evader, 0 none) when a failure decides the game: with
  rules.failure_loses_game the side that alone failed loses; with rules.pursuer_out_wins only a lone
  pursuer arena exit does (an evader win)."""
  t, rules = env.termination_manager.get_term, tag_rules(env)
  robot_fail, opp_fail = t("fell_over") | t("robot_out"), t("opponent_fell_over") | t("opponent_out")
  pursuer = role(env) > 0
  p_fail, e_fail = torch.where(pursuer, robot_fail, opp_fail), torch.where(pursuer, opp_fail, robot_fail)
  p_out = torch.where(pursuer, t("robot_out"), t("opponent_out"))
  lone_p, lone_e = p_fail & ~e_fail, e_fail & ~p_fail
  if rules.failure_loses_game:
    return lone_e.float() - lone_p.float()
  return -(lone_p & p_out & rules.pursuer_out_wins).float()


def outcome(env: ManagerBasedRlEnv, learner_role: torch.Tensor) -> torch.Tensor:
  if tag_rules(env).post_tag_s > 0:
    return env.command_manager.get_term("role").game_result
  t = env.termination_manager.get_term
  return game_outcome(t("tagged"), t("time_out"), failed(env), learner_role, forfeit(env))


def tag_result(env: ManagerBasedRlEnv) -> torch.Tensor:
  if tag_rules(env).post_tag_s > 0:
    state = env.command_manager.get_term("role")
    # The +/-10 payoff is paid at first contact, NOT again at recovery completion.
    pre_tag = state.tag_step < 0
    return game_outcome(state.tag_pulse, env.termination_manager.get_term("time_out"),
                        failed(env) & pre_tag, role(env), forfeit(env) * pre_tag)
  return outcome(env, role(env))


def own_failure(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Individual penalty, mirrored per seat (NOT a zero-sum game reward)."""
  t, rules = env.termination_manager.get_term, tag_rules(env)
  fall, out = ("fell_over", "robot_out") if asset_cfg.name == "robot" else ("opponent_fell_over", "opponent_out")
  penalty = (t(fall) & rules.fall_loses) | (t(out) & rules.out_loses)
  if rules.post_tag_s > 0 and rules.post_tag_failure == "pursuer":
    # The tagged evader already lost 10; a second exposure makes pre-TAG self-falls relatively cheap.
    pursuer = role(env) > 0 if asset_cfg.name == "robot" else role(env) < 0
    penalty &= pursuer | (env.command_manager.get_term("role").tag_step < 0)
  if not rules.failure_time_penalty:
    return -penalty.float()
  # Pre-TAG failures also forfeit the remaining game time, so falling early never undercuts playing
  # on while every upright step costs something (E032: an -18 timeout loss vs -20 for falling at once).
  pre_tag = env.command_manager.get_term("role").tag_step < 0
  left = (rules.episode_s - env.episode_length_buf * env.step_dt).clamp_min(0)
  return -penalty.float() * (1 + pre_tag * rules.failure_time_penalty * left / rules.win_reward)


def own_inward_speed(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  """Own planar velocity projected toward the other robot, not relative closing speed."""
  other = "opponent" if name == "robot" else "robot"
  a, b = env.scene[name].data, env.scene[other].data
  delta = b.root_link_pos_w[:, :2] - a.root_link_pos_w[:, :2]
  return (a.root_link_lin_vel_w[:, :2] * delta).sum(-1) / delta.norm(dim=-1).clamp_min(1e-6)


def contact_speed_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Bounded, individual first-contact regularizer; TAG payoff itself is unchanged.

  Same rule for either role: a stationary or fleeing agent is not charged for
  the other robot's approach. Charged once, on the step of the first valid TAG.
  """
  rules, state = tag_rules(env), env.command_manager.get_term("role")
  index = 0 if asset_cfg.name == "robot" else 1
  speed = state.inward_speed[:, index]
  cost = (rules.contact_speed_weight * (speed - rules.contact_speed_limit_mps).clamp_min(0).square()).clamp_max(rules.contact_speed_max_penalty)
  valid_tag = state.tag_pulse if rules.post_tag_s > 0 else env.termination_manager.get_term("tagged") & ~failed(env)
  return -cost * valid_tag


def tilt_spin(env: ManagerBasedRlEnv, name: str) -> tuple[torch.Tensor, torch.Tensor]:
  """Root tilt from vertical (rad) and root angular speed (rad/s)."""
  d = env.scene[name].data
  q = d.root_link_quat_w
  return torch.acos((1 - 2 * (q[:, 1].square() + q[:, 2].square())).clamp(-1, 1)), d.root_link_ang_vel_w.norm(dim=-1)


def tag_posture_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Bounded, individual cost of the pursuer's own spin/tilt at the first valid TAG.

  Spin at TAG best predicts a fall in the following second; the evader is never charged.
  """
  rules = tag_rules(env)
  tilt, spin = tilt_spin(env, asset_cfg.name)
  excess = ((spin / rules.tag_spin_limit_radps - 1).clamp_min(0).square()
            + (torch.rad2deg(tilt) / rules.tag_tilt_limit_deg - 1).clamp_min(0).square())
  cost = (rules.tag_posture_weight * excess).clamp_max(rules.tag_posture_max_penalty)
  pursuer = role(env) > 0 if asset_cfg.name == "robot" else role(env) < 0
  valid_tag = (env.command_manager.get_term("role").tag_pulse if rules.post_tag_s > 0
               else env.termination_manager.get_term("tagged") & ~failed(env))
  return -cost * (valid_tag & pursuer)


def motion_quality(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  """Physical diagnostics only; these do not modify actions or rewards."""
  a, b = env.scene["robot"].data, env.scene["opponent"].data
  p = role(env) > 0
  (tilt_a, spin_a), (tilt_b, spin_b) = tilt_spin(env, "robot"), tilt_spin(env, "opponent")
  tilt, angular = torch.where(p, tilt_a, tilt_b), torch.where(p, spin_a, spin_b)
  delta, velocity = b.root_link_pos_w[:, :2] - a.root_link_pos_w[:, :2], b.root_link_lin_vel_w[:, :2] - a.root_link_lin_vel_w[:, :2]
  closing = -(delta * velocity).sum(-1) / delta.norm(dim=-1).clamp_min(1e-6)
  force = torch.zeros_like(closing)
  if tag_rules(env).tag_mode == "contact":
    force = torch.stack([env.scene[name].data.force_history.norm(dim=-1).flatten(1).max(-1).values
                         for name in ("tag_robot", "tag_opponent")]).max(0).values
  height = [env.scene[n].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] for n in ("robot", "opponent")]
  arms = [arm_pose_error(env, n) for n in ("robot", "opponent")]
  return {"distance_m": delta.norm(dim=-1), "closing_speed_mps": closing,
          "pursuer_root_tilt_deg": torch.rad2deg(tilt), "pursuer_angular_speed_radps": angular,
          "interagent_peak_force_N": force,
          "pursuer_pelvis_height_m": torch.where(p, *height), "evader_pelvis_height_m": torch.where(p, height[1], height[0]),
          "pursuer_arm_pose_error": torch.where(p, *arms), "evader_arm_pose_error": torch.where(p, arms[1], arms[0])}


def arm_pose_error(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  """Mean squared arm-joint deviation from the default pose, in units of ARM_POSE_STD (1 = one std)."""
  asset = env.scene[name]
  ids, names = asset.find_joints(tuple(ARM_POSE_STD))
  _, _, std = resolve_matching_names_values(ARM_POSE_STD, names)
  err = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
  return (err / torch.tensor(std, device=env.device)).square().mean(-1)


def quality_metric(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  return motion_quality(env)[name]


def ending_metric(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  t = env.termination_manager.get_term
  hit = (env.command_manager.get_term("role").tag_step >= 0) if tag_rules(env).post_tag_s > 0 else t("tagged") & ~failed(env)
  flags = {"valid_tag": hit,
           "healthy_timeout": t("time_out") & ~hit & ~failed(env), "physical_failure": failed(env),
           "post_tag_failure": hit & failed(env), "post_tag_survived": hit & t("tagged") & ~failed(env)}
  if tag_rules(env).post_tag_s > 0:  # Only worlds that actually ran a recovery window.
    window = torch.as_tensor(recovery_s(env), device=env.device) > 0
    flags["post_tag_failure"] &= window
    flags["post_tag_survived"] &= window
  return flags[name].float()


def reward_component(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  """True episode sums, unlike stock Episode_Reward's per-max-duration rates."""
  manager = env.reward_manager
  if name not in manager.active_terms:
    return torch.zeros(env.num_envs, device=env.device)
  scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
  return manager._step_reward[:, manager.active_terms.index(name)] * scale


def upright_offset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """Constant per-second offset; paired with upright, never a change to TAG payoff. With
  upright_offset_roles=pursuer only the pursuer seat pays it: a charged evader that cannot escape
  scores higher by being tagged early (every step <= 0), so it stops fleeing even when it could."""
  if tag_rules(env).upright_offset_roles == "both":
    return torch.ones(env.num_envs, device=env.device)
  return (role(env) > 0 if asset_cfg.name == "robot" else role(env) < 0).float()


# Locomotion running arm stds (mjlab G1 velocity task): room for arm swing while running.
ARM_POSE_STD = {r".*shoulder_pitch.*": 0.5, r".*shoulder_roll.*": 0.2, r".*shoulder_yaw.*": 0.15,
                r".*elbow.*": 0.35, r".*wrist.*": 0.3}


BODY_POSE_STD = {r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.2, r".*hip_yaw.*": 0.2, r".*knee.*": 0.6,
                 r".*ankle_pitch.*": 0.35, r".*ankle_roll.*": 0.15, r".*waist_yaw.*": 0.3, r".*waist_roll.*": 0.08,
                 r".*waist_pitch.*": 0.2, **ARM_POSE_STD}


class arm_pose_cost(mdp_posture):
  """1 - posture kernel over the arm joints: 0 at the default pose, 1 far from it. The TAG task
  drops the command-gated locomotion pose term, which otherwise leaves the arms unregularized."""

  def __call__(self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return 1 - super().__call__(env, std, asset_cfg)


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
  if rules.upright_reward_offset:
    cfg.rewards["upright_offset"] = RewardTermCfg(
      func=upright_offset, weight=-rules.upright_reward_offset * cfg.rewards["upright"].weight
    )
  if rules.arm_pose_weight:
    arms = SceneEntityCfg("robot", joint_names=tuple(ARM_POSE_STD))
    cfg.rewards["arm_pose"] = RewardTermCfg(func=arm_pose_cost, weight=-rules.arm_pose_weight,
                                            params={"asset_cfg": arms, "std": ARM_POSE_STD})
  if rules.body_pose_weight:  # Same kernel, every joint.
    body = SceneEntityCfg("robot", joint_names=tuple(BODY_POSE_STD))
    cfg.rewards["body_pose"] = RewardTermCfg(func=arm_pose_cost, weight=-rules.body_pose_weight,
                                             params={"asset_cfg": body, "std": BODY_POSE_STD})
  cfg.rewards["tag"] = RewardTermCfg(func=tag_result, weight=rules.win_reward / dt)
  cfg.rewards["failure"] = RewardTermCfg(func=own_failure, weight=rules.win_reward / dt,
                                       params={"asset_cfg": SceneEntityCfg("robot")})
  if rules.tag_mode == "contact":
    cfg.rewards["contact_speed"] = RewardTermCfg(func=contact_speed_penalty, weight=1.0 / dt,
                                                params={"asset_cfg": SceneEntityCfg("robot")})
  if rules.tag_posture_weight:
    cfg.rewards["tag_posture"] = RewardTermCfg(func=tag_posture_penalty, weight=1.0 / dt,
                                              params={"asset_cfg": SceneEntityCfg("robot")})
  if shaping_cfg.enabled:
    cfg.commands["potential"] = PotentialCfg(potential=shaping_cfg.potential)  # After role.
    cfg.rewards["shaping"] = RewardTermCfg(
      func=shaping, weight=shaping_cfg.weight / dt,
      params={"gamma": shaping_cfg.gamma, "name": shaping_cfg.potential},
    )
  cfg.metrics.update({f"reward_{name}_episode_sum": MetricsTermCfg(
    func=reward_component, params={"name": name}, reduce="sum")
    for name in ("tag", "shaping", "failure", "contact_speed", "opp_contact_speed", "tag_posture", "opp_tag_posture",
                 "upright", "upright_offset")})
  if rules.tag_mode == "contact":  # Pursuer's tag_bodies subtrees vs. the other's whole body.
    cfg.scene.sensors += tuple(
      ContactSensorCfg(
        name=f"tag_{me}",
        primary=ContactMatch(mode=rules.tag_match, pattern=rules.tag_bodies, entity=me),
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
