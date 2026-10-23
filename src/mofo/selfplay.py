"""Symmetric self-play: the learner also plays the opponent seat. Host side only.

``selfplay_env_cfg`` mirrors everything the learner seat has onto the opponent: the robot-only
sensors, a critic group (``opponent_critic``), the locomotion regularizers (``opp_*`` reward
terms) and the startup domain randomization. ``SeatRewards`` splits mjlab's per-term step
rewards into one reward per seat; the zero-sum terms (tag, shaping, clock) enter the opponent seat
negated, so phi_opp = -phi_robot keeps the shaping potential-based for both seats.
"""

import inspect
from dataclasses import replace

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg

from mofo.tasks.tag import ACTION, tag_features

PREFIX = "opp_"
ROBOT_SENSORS = ("foot_height_scan", "feet_ground_contact", "self_collision")  # Not entity-prefixed.
ZERO_SUM = ("tag", "shaping", "clock")  # Robot-seat terms whose opponent-seat value is the negative.


def _sensor(name: str) -> str:
  return PREFIX + name if name in ROBOT_SENSORS else name.replace("robot/", "opponent/")


def mirror_params(params: dict, func) -> dict:
  """Term params pointed at the opponent: entity, sensors, action term, tag features."""
  p = dict(params)
  for k in ("sensor_name", "height_sensor_name"):
    if k in p:
      p[k] = _sensor(p[k])
  if "action_name" in p:
    p["action_name"] = ACTION["opponent"]
  if p.get("command_name") == "twist":  # Gait terms' stand-in command: the opponent's own velocity.
    p["command_name"] = PREFIX + "twist"
  if func is tag_features:
    p.update(me="opponent", other="robot", role_sign=-1.0)
  elif "asset_cfg" in p or "asset_cfg" in inspect.signature(func.__call__ if inspect.isclass(func) else func).parameters:
    p["asset_cfg"] = replace(p.get("asset_cfg", SceneEntityCfg("robot")), name="opponent")
  return p


def _mirror_sensor(s):
  s = replace(s, name=PREFIX + s.name)
  if isinstance(s, ContactSensorCfg):
    swap = lambda m: m if m is None or m.entity != "robot" else replace(m, entity="opponent")
    return replace(s, primary=swap(s.primary), secondary=swap(s.secondary))
  return replace(s, frame=tuple(replace(f, entity="opponent") for f in s.frame))


def selfplay_env_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Add the opponent seat's sensors, critic obs, regularizers and startup DR."""
  cfg.scene.sensors += tuple(_mirror_sensor(s) for s in cfg.scene.sensors if s.name in ROBOT_SENSORS)
  critic = cfg.observations["critic"]
  cfg.observations["opponent_critic"] = replace(
    critic, terms={k: replace(t, params=mirror_params(t.params, t.func)) for k, t in critic.terms.items()}
  )
  for k, t in list(cfg.rewards.items()):
    if k not in ZERO_SUM:
      cfg.rewards[PREFIX + k] = replace(t, params=mirror_params(t.params, t.func))
  if "twist" in cfg.commands:
    cfg.commands[PREFIX + "twist"] = replace(cfg.commands["twist"], entity="opponent")
  for k, e in list(cfg.events.items()):
    if e.mode == "startup":
      cfg.events[PREFIX + k] = replace(e, params=mirror_params(e.params, e.func))
  return cfg


class SeatRewards:
  """Per-seat step rewards from mjlab's per-term step rewards (the RewardManager sums all
  terms into one buffer). Robot seat: every non-``opp_*`` term. Opponent seat: the ``opp_*``
  terms minus the zero-sum terms. Each regularizer counts for exactly one seat."""

  def __init__(self, env: ManagerBasedRlEnv):
    names = env.reward_manager.active_terms
    self.opp = torch.tensor([n.startswith(PREFIX) for n in names], device=env.device)
    self.zero_sum = torch.tensor([n in ZERO_SUM for n in names], device=env.device)
    self.scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0

  def __call__(self, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    r = env.reward_manager._step_reward * self.scale  # weight * value per term; as reward_buf.
    return r[:, ~self.opp].sum(-1), r[:, self.opp].sum(-1) - r[:, self.zero_sum].sum(-1)
