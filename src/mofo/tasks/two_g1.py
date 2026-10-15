"""Two Unitree G1 humanoids in one flat-ground scene."""

import math
from dataclasses import replace

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

ROBOTS = {"g1_a": (-1.0, 0.0), "g1_b": (1.0, 0.0)}  # Slot name -> (x, y) start.


def _robot(x: float, y: float):
  cfg = get_g1_robot_cfg()
  cfg.init_state = replace(cfg.init_state, pos=(x, y, cfg.init_state.pos[2]))
  return cfg


def _obs(name: str) -> ObservationGroupCfg:
  asset = {"asset_cfg": SceneEntityCfg(name)}
  terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, params=asset),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, params=asset),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, params=asset),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params=asset),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params=asset),
    "actions": ObservationTermCfg(func=mdp.last_action, params={"action_name": name}),
  }
  return ObservationGroupCfg(terms=terms, concatenate_terms=True)


def two_g1_env_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  """Two G1s 2 m apart on x; one action term and one observation group per slot."""
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=5.0,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={name: _robot(*xy) for name, xy in ROBOTS.items()},
    ),
    actions={
      name: JointPositionActionCfg(
        entity_name=name,
        actuator_names=(".*",),
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
      )
      for name in ROBOTS
    },
    observations={name: _obs(name) for name in ROBOTS},
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      **{
        f"{name}_fell_over": TerminationTermCfg(
          func=mdp.bad_orientation,
          params={"limit_angle": math.radians(70.0), "asset_cfg": SceneEntityCfg(name)},
        )
        for name in ROBOTS
      },
    },
    sim=SimulationCfg(
      njmax=600, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)
    ),
    decimation=4,
    episode_length_s=20.0,
  )
