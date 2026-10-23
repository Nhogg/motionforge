"""Tag env construction from a Hydra config: shared by scripts/mofo_train.py and scripts/eval.py."""

import math
from dataclasses import replace
from types import SimpleNamespace

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.camera_sensor import CameraSensorCfg
from mjlab.terrains.terrain_entity import TerrainEntity
from omegaconf import DictConfig, OmegaConf

from mofo.selfplay import selfplay_env_cfg
from mofo.tasks.tag import GaitCfg, ResetCfg, ShapingCfg, TagRules, tag_env_cfg

GAME_SCHEMAS = {"rules": TagRules, "reset": ResetCfg, "shaping": ShapingCfg, "gait": GaitCfg}  # Typed Hydra groups.


def game(cfg: DictConfig, **overrides) -> SimpleNamespace:
  """The typed rules/reset/shaping groups as dataclasses, with field overrides per group."""
  g = {k: OmegaConf.to_object(cfg[k]) if k in cfg else s() for k, s in GAME_SCHEMAS.items()}  # Missing: defaults.
  return SimpleNamespace(**{k: replace(v, **overrides.get(k, {})) for k, v in g.items()})


def make_env(cfg, num_envs, g: SimpleNamespace, eval=False, video=False, selfplay=False) -> ManagerBasedRlEnv:
  """``video``: robots tinted by role (cfg.video.*_rgba; tag only) and a world camera (cfg.video.camera)
  looking at env 0's arena center from ``back`` m behind (-y) and ``up`` m high."""
  goal = cfg.get("task", "tag") == "goal"  # One G1, no opponent.
  if goal:
    from mofo.tasks.goal import GoalCfg, goal_env_cfg

    env_cfg = goal_env_cfg(g.rules, g.reset, GoalCfg(**OmegaConf.to_container(cfg.goal)) if "goal" in cfg else GoalCfg())
  else:
    env_cfg = tag_env_cfg(g.rules, g.reset, g.shaping, g.gait)
    if selfplay:
      selfplay_env_cfg(env_cfg)
  env_cfg.scene.num_envs, env_cfg.seed, env_cfg.scene.env_spacing = num_envs, cfg.seed, cfg.env_spacing
  if eval:
    for grp in ("actor",) if goal else ("actor", "opponent"):
      env_cfg.observations[grp].enable_corruption = False
  if video:
    v, c = cfg.video, cfg.video.camera
    if not goal:
      env_cfg.commands["role"] = replace(env_cfg.commands["role"], colors=(tuple(v.pursuer_rgba), tuple(v.evader_rgba)))
    o = TerrainEntity._compute_env_origins_grid(SimpleNamespace(_device="cpu"), num_envs, env_cfg.scene.env_spacing)[0]
    a = math.atan2(c.back, c.up)
    env_cfg.scene.sensors += (CameraSensorCfg(
      name="world_cam", pos=(o[0].item(), o[1].item() - c.back, c.up), quat=(math.cos(a / 2), math.sin(a / 2), 0.0, 0.0),
      width=c.width, height=c.height, fovy=c.fovy, data_types=tuple(c.data_types),
    ),)
  return ManagerBasedRlEnv(env_cfg, device=cfg.device)
