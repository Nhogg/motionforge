"""Stage-2 tag fine-tuning, evaluation and env checks for the two-G1 tag task.

Hydra config: config/mofo_train.yaml; cmd=tag|eval|check (or a bare first word) picks
config/cmd/, and the typed groups rules/, reset/, shaping/ hold the game (mofo.tasks.tag).

  uv run scripts/mofo_train.py tag cmd.init=STAGE1.pt [shaping.enabled=true] [reset=legacy] [agent.num_mini_batches=8] [league.selfplay_frac=0.5 league.snapshot_every=250]
  uv run scripts/mofo_train.py eval num_envs=512 cmd.policy=CKPT [cmd.baseline=true] [cmd.video=OUT.mp4]
  uv run scripts/mofo_train.py check [rules.tag_mode=contact] [cmd.plots=DIR]
"""

import itertools
import json
import math
import os
from pathlib import Path
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "disable")  # Before mjlab import (it defaults to egl).

import torch
from hydra.core.hydra_config import HydraConfig
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.camera_sensor import CameraSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.terrains.terrain_entity import TerrainEntity
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from omegaconf import DictConfig, OmegaConf
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mofo.cli import run
from mofo.checkpoint_video import CheckpointVideoRunner, evaluation_config
from mofo.league import SCRIPTED, SCRIPTED_ID, FixedLeague, League, LeagueCfg, Opponent
from mofo.opponent import OpponentEnv, ScriptedPolicy, load_policy
from mofo.selfplay import PREFIX, ZERO_SUM, selfplay_env_cfg
from mofo.evaluation import TagTrials, summarize_trials
from mofo.role_model import RoleModel
from mofo.tasks.tag import POTENTIALS, ResetCfg, ShapingCfg, TagRules, distance, outcome, role, tag_env_cfg, tagged, failed, motion_quality, wall_distance, tag_observation_dim

_STOCK = unitree_g1_ppo_runner_cfg()


@dataclass(frozen=True)
class AgentCfg:
  """PPO settings for cmd=tag: stock mjlab G1 (rl_cfg.py) except the fine-tune lr."""

  num_steps_per_env: int = _STOCK.num_steps_per_env
  num_mini_batches: int = _STOCK.algorithm.num_mini_batches
  num_learning_epochs: int = _STOCK.algorithm.num_learning_epochs
  learning_rate: float = 1.0e-4  # Fresh optimizer (stage 1 starts at 1e-3).
  schedule: str = _STOCK.algorithm.schedule  # adaptive: KL-driven lr.
  desired_kl: float = _STOCK.algorithm.desired_kl
  gamma: float = _STOCK.algorithm.gamma
  lam: float = _STOCK.algorithm.lam
  entropy_coef: float = _STOCK.algorithm.entropy_coef
  clip_param: float = _STOCK.algorithm.clip_param
  max_grad_norm: float = _STOCK.algorithm.max_grad_norm
  save_interval: int = _STOCK.save_interval
  separate_roles: bool = False  # Independent actor/critic parameters and normalization per role.
  max_learning_rate: float | None = None  # Ceiling for the adaptive schedule (rsl-rl default 1e-2).


SCHEMAS = {"rules": TagRules, "reset": ResetCfg, "shaping": ShapingCfg, "agent": AgentCfg, "league": LeagueCfg}


def game(cfg: DictConfig, **overrides) -> SimpleNamespace:
  """The typed rules/reset/shaping groups as dataclasses, with field overrides per group."""
  g = {k: OmegaConf.to_object(cfg[k]) for k in ("rules", "reset", "shaping")}
  return SimpleNamespace(**{k: replace(v, **overrides.get(k, {})) for k, v in g.items()})


def make_env(cfg, num_envs, g: SimpleNamespace, eval=False, video=False, selfplay=False) -> ManagerBasedRlEnv:
  # Evaluation owns its independent first-TAG/recovery audit; never combine it
  # with the training grace phase or shorten a late TAG's audit at the game limit.
  env_cfg = tag_env_cfg(replace(g.rules, post_tag_s=0.0) if eval else g.rules, g.reset, g.shaping)
  if selfplay:
    selfplay_env_cfg(env_cfg)
  env_cfg.scene.num_envs, env_cfg.seed, env_cfg.scene.env_spacing = num_envs, cfg.seed, cfg.env_spacing
  if eval:
    for g in ("actor", "opponent"):
      env_cfg.observations[g].enable_corruption = False
  if eval and 'post_tag_seconds' in cfg.cmd:
    # Capture the actual end state, then continue TAGs for a recovery audit.
    env_cfg.auto_reset = False
    env_cfg.terminations['tagged'] = replace(env_cfg.terminations['tagged'], func=lambda e: torch.zeros(e.num_envs, device=e.device, dtype=torch.bool))
    env_cfg.episode_length_s += cfg.cmd.post_tag_seconds
    if cfg.cmd.post_tag_ignore_oob:
      recovery_mask = torch.zeros(num_envs, dtype=torch.bool, device=cfg.device)
      for name in ('robot_out', 'opponent_out'):
        term = env_cfg.terminations[name]
        def boundary_before_tag(e, func=term.func, **kwargs):
          return func(e, **kwargs) & ~recovery_mask
        env_cfg.terminations[name] = replace(term, func=boundary_before_tag)
  if video:  # World camera on env 0's arena; robots tinted by role.
    env_cfg.commands["role"] = replace(env_cfg.commands["role"], colors=(tuple(cfg.video.pursuer_rgba), tuple(cfg.video.evader_rgba)))
    o = TerrainEntity._compute_env_origins_grid(SimpleNamespace(_device="cpu"), num_envs, env_cfg.scene.env_spacing)[0]
    a = math.atan2(9.0, 7.0)  # Look from 9 m behind, 7 m up, at the arena center.
    env_cfg.scene.sensors += (CameraSensorCfg(
      name="world_cam", pos=(o[0].item(), o[1].item() - 9.0, 7.0), quat=(math.cos(a / 2), math.sin(a / 2), 0.0, 0.0),
      width=640, height=480,
    ),)
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  if eval and cfg.cmd.get('post_tag_ignore_oob', False):
    env._post_tag_recovery = recovery_mask
  return env


def make_league(cfg, env, lc: LeagueCfg, ckpt) -> League:
  """Training pool: the scripted stage-1 opponent (never evicted), cfg.init_pool, and learner snapshots."""
  league = League(lc, seed=cfg.seed, load=lambda p: load_policy(p, cfg.device, env), snapshot_dir=os.path.join(cfg.cmd.logdir, "league"))
  league.add(Opponent(SCRIPTED_ID, ScriptedPolicy(env, load_policy(ckpt, cfg.device) if ckpt else None), SCRIPTED, ckpt))
  return league


def scripted_league(env, ckpt, device) -> FixedLeague:
  policy = load_policy(ckpt, device) if ckpt else None
  return FixedLeague(Opponent(f"scripted:{ckpt or 'zero'}", ScriptedPolicy(env, policy)))


def warm_start(runner, ckpt: str, env: ManagerBasedRlEnv) -> None:
  """Load stage-1 actor/critic into the tag models, mapping first-layer columns by term.

  Shared terms keep their weights and normalizer stats; new tag terms get zero columns.
  The dropped command input is folded into the first-layer bias as a zero command, so
  the warm-started policy equals stage 1 commanded to stand still.
  """
  sd = torch.load(ckpt, map_location=runner.device, weights_only=False)
  om, old_cfg = env.observation_manager, unitree_g1_flat_env_cfg().observations
  for group, model in (("actor", runner.alg.actor), ("critic", runner.alg.critic)):
    old, new = sd[f"{group}_state_dict"], {k: v.clone() for k, v in model.state_dict().items()}
    dims = {k: d[0] for k, d in zip(om.active_terms[group], om.group_obs_term_dim[group])}
    names = list(old_cfg[group].terms)
    (dropped,) = [k for k in names if k not in dims]  # The velocity command.
    dims_old = {**dims, dropped: old["mlp.0.weight"].shape[1] - sum(dims.get(k, 0) for k in names)}
    starts = dict(zip(names, torch.tensor([0] + [dims_old[k] for k in names]).cumsum(0).tolist()))
    src, dst, off = [], [], 0
    for k in om.active_terms[group]:
      if k in starts:
        src += range(starts[k], starts[k] + dims[k])
        dst += range(off, off + dims[k])
      off += dims[k]
    cmd = list(range(starts[dropped], starts[dropped] + dims_old[dropped]))
    w = old["mlp.0.weight"]
    new["mlp.0.weight"].zero_()[:, dst] = w[:, src]
    new["mlp.0.bias"] = old["mlp.0.bias"].clone()
    if "obs_normalizer._mean" in old:
      mean, std = old["obs_normalizer._mean"][0, cmd], old["obs_normalizer._std"][0, cmd]
      new["mlp.0.bias"] += w[:, cmd] @ (-mean / (std + 1e-2))  # eps of EmpiricalNormalization.
      for k in ("_mean", "_var", "_std"):
        new[f"obs_normalizer.{k}"][:, dst] = old[f"obs_normalizer.{k}"][:, src]
      new["obs_normalizer.count"] = old["obs_normalizer.count"]
    for k, v in old.items():
      if k not in ("mlp.0.weight", "mlp.0.bias") and not k.startswith("obs_normalizer."):
        new[k] = v
    model.load_state_dict(new)
    if isinstance(model, RoleModel):
      model.evader.load_state_dict({k: v for k, v in new.items()
                                   if not k.startswith("evader.") and k != "_role_column"})
  print(f"[warm start] loaded {ckpt}; dropped '{dropped}' columns, zero-init new tag columns")


def configure_logging(rl, cfg: DictConfig) -> dict:
  """Use rsl-rl's shared W&B/TensorBoard writer; keep the resolved task config."""
  if cfg.cmd.logger not in ("tensorboard", "wandb"):
    raise ValueError("cmd.logger must be tensorboard or wandb")
  rl.logger, rl.upload_model = cfg.cmd.logger, False
  rl.run_name = os.path.basename(os.path.normpath(cfg.cmd.logdir))
  if cfg.cmd.logger == "wandb":
    wb = cfg.cmd.wandb
    if wb.mode not in ("online", "offline", "disabled"):
      raise ValueError("cmd.wandb.mode must be online, offline or disabled")
    os.environ["WANDB_MODE"] = wb.mode
    if wb.entity:
      # rsl-rl's writer reads WANDB_USERNAME and passes it as wandb.init(entity=...).
      os.environ["WANDB_USERNAME"] = wb.entity
    if wb.group:
      os.environ["WANDB_RUN_GROUP"] = wb.group
    rl.logger = {"class_name": "mofo.logging:WandbWriter", "project_name": wb.project}
    rl.upload_model = wb.upload_model
  return {**asdict(rl), "motionforge": OmegaConf.to_container(cfg, resolve=True)}


def train(cfg: DictConfig) -> None:
  """Runs in its logdir (the Hydra run dir), so <logdir>/.hydra/ records the full config."""
  if cfg.cmd.resume:
    if cfg.cmd.init:
      raise ValueError('cmd.init is stage1-only; use cmd.resume for TAG and opponent=STAGE1.pt')
    if not cfg.opponent:
      raise ValueError('TAG resume requires opponent=STAGE1.pt for the scripted league/render reference')
    if Path(cfg.cmd.resume).resolve().parent == Path(cfg.cmd.logdir).resolve():
      raise ValueError('Resume into a new logdir to preserve the source checkpoint and configuration')
  g, agent = game(cfg), OmegaConf.to_object(cfg.agent)
  agent = replace(agent, **{k: v for k, v in (("learning_rate", cfg.cmd.lr), ("save_interval", cfg.cmd.save_interval)) if v is not None})
  rl = unitree_g1_ppo_runner_cfg()
  for k, v in asdict(agent).items():
    if k not in ("separate_roles", "max_learning_rate"):
      setattr(rl.algorithm if hasattr(rl.algorithm, k) else rl, k, v)
  assert g.shaping.gamma == rl.algorithm.gamma, f"shaping.gamma {g.shaping.gamma} != PPO gamma {rl.algorithm.gamma}"
  lc: LeagueCfg = OmegaConf.to_object(cfg.league)
  rl.seed, rl.max_iterations = cfg.seed, cfg.cmd.iterations
  train_cfg = configure_logging(rl, cfg)  # Validate before allocating the simulator.
  if cfg.cmd.render.enabled:
    evaluation_config(train_cfg["motionforge"], os.path.join(cfg.cmd.logdir, "model_0.pt"))
  env = make_env(cfg, cfg.num_envs, g, selfplay=lc.selfplay_frac > 0)
  league = make_league(cfg, env, lc, cfg.opponent or cfg.cmd.init)
  venv = OpponentEnv(env, league, steps_per_iter=rl.num_steps_per_env)
  rl.experiment_name = ("tag_shaping" if g.shaping.enabled else "tag") + ("_selfplay" if venv.n_sp else "") + ("_league" if lc.snapshot_every or lc.init_pool else "")
  train_cfg["experiment_name"] = rl.experiment_name
  if agent.max_learning_rate is not None:
    if agent.max_learning_rate < agent.learning_rate:
      raise ValueError("agent.max_learning_rate must be >= agent.learning_rate")
    train_cfg["algorithm"].update(class_name="mofo.ppo:CappedPPO", max_learning_rate=agent.max_learning_rate)
  if agent.separate_roles:
    om = env.observation_manager
    for group in ("actor", "critic"):
      index = om.active_terms[group].index("tag")
      column = sum(d[0] for d in om.group_obs_term_dim[group][:index]) + 4
      train_cfg[group].update(class_name="mofo.role_model:RoleModel", role_column=column)
  runner = CheckpointVideoRunner(venv, train_cfg, cfg.cmd.logdir, cfg.device)
  venv.learner = runner.alg.actor  # Frozen into the league every league.snapshot_every iterations.
  if cfg.cmd.resume:
    runner.resume(cfg.cmd.resume, restore_league=cfg.cmd.resume_league,
                  restore_optimizer=cfg.cmd.resume_optimizer, evader_path=cfg.cmd.resume_evader)
  elif cfg.cmd.init:
    warm_start(runner, cfg.cmd.init, env)
    obs = venv.get_observations().to(cfg.device)  # Must match stage 1 under a zero command.
    proprio = obs["actor"][:, :-tag_observation_dim(env)]
    ref = load_policy(cfg.cmd.init, cfg.device)(torch.cat([proprio, proprio.new_zeros(len(proprio), 3)], -1))
    err = (runner.alg.actor(obs) - ref).abs().max().item()
    print(f"[warm start] max |a_tag - a_stage1(cmd=0)| = {err:.2e}")
    assert err < 1e-3
  # Fixed start lengths: the timeout decides the game, so don't randomize episode length.
  runner.save(os.path.join(cfg.cmd.logdir, 'initial.pt'))  # Before any PPO updates.
  runner.learn(cfg.cmd.iterations, init_at_random_ep_len=False)


def evaluate(cfg: DictConfig) -> None:
  if not math.isfinite(cfg.cmd.scripted_speed_scale) or not 0 < cfg.cmd.scripted_speed_scale <= 1:
    raise ValueError('cmd.scripted_speed_scale must be finite and in (0, 1]')
  if cfg.cmd.opponent_policy and cfg.cmd.scripted_speed_scale != 1:
    raise ValueError('Scripted speed scaling cannot be applied to a learned opponent')
  g = game(cfg)
  env = make_env(cfg, cfg.num_envs, g, eval=True, video=bool(cfg.cmd.video))
  if cfg.cmd.opponent_policy:  # Head-to-head: a tag checkpoint acts on the opponent obs group directly.
    p = cfg.cmd.opponent_policy
    league = FixedLeague(Opponent(f"learner:{p}", load_policy(p, cfg.device, env), "historical", p))
  else:
    league = scripted_league(env, cfg.opponent, cfg.device)
    league.opponent.policy.speed_scale = cfg.cmd.scripted_speed_scale
  venv = OpponentEnv(env, league)
  policy = load_policy(cfg.cmd.policy, cfg.device, None if cfg.cmd.baseline else env) if cfg.cmd.policy else None
  learner = ScriptedPolicy(env, policy, me="robot") if cfg.cmd.baseline or policy is None else policy
  robot = env.scene["robot"].data
  if cfg.cmd.post_tag_seconds < 0:
    raise ValueError('post_tag_seconds must be nonnegative')
  trials = TagTrials(env.num_envs, env.device, env.step_dt, g.rules.episode_s, cfg.cmd.post_tag_seconds, distance(env))
  speed, yaw_rate, frames = [], [], []
  # Own-seat motor costs only: evaluation suppresses TAG termination, so its
  # reward buffer is NOT a faithful total game return. Keep these separate.
  names = env.reward_manager.active_terms
  motor_names = [k for k in names if k not in ('tag', 'shaping', 'failure', 'contact_speed') and not k.startswith('opp_')]
  motor_columns = [names.index(k) for k in motor_names]
  motor_sums = torch.zeros((env.num_envs, len(motor_names)), device=env.device)
  motor_discounted = torch.zeros(env.num_envs, device=env.device)
  path_length = torch.zeros_like(motor_discounted)
  initial_xy = robot.root_link_pos_w[:, :2].clone()
  obs = venv.get_observations()
  by_second = {1.0: [], -1.0: []}  # Learner role -> mean robot distance at each whole second.
  with torch.no_grad():
    for t in itertools.count():  # Each env's first episode, so short ones aren't over-sampled.
      if trials.finished.all():
        break
      r = role(env).clone()
      d = distance(env)
      if t % round(1 / env.step_dt) == 0:
        for k in by_second:
          m = (trials.cause == 0) & (r == k)
          by_second[k].append(round(d[m].mean().item(), 3) if m.any() else None)
      live = trials.cause == 0
      planar_speed = robot.root_link_lin_vel_w[:, :2].norm(dim=-1)
      speed.append(planar_speed[live].clone())
      path_length += planar_speed * env.step_dt * live
      yaw_rate.append(robot.root_link_ang_vel_w[:, 2].abs()[live].clone())
      if cfg.cmd.post_tag_ignore_oob:
        env._post_tag_recovery.copy_(trials.cause == 1)
      obs, _, dones, _ = venv.step(learner(obs["actor"]))
      tm = env.termination_manager.get_term
      failure, physics_done = failed(env), dones.bool()
      scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
      motor_step = env.reward_manager._step_reward[:, motor_columns] * scale * live[:, None]
      motor_sums += motor_step
      motor_discounted += cfg.agent.gamma**t * motor_step.sum(-1)
      quality = {**motion_quality(env),
                 'learner_path_length_m': path_length,
                 'learner_displacement_m': (robot.root_link_pos_w[:, :2] - initial_xy).norm(dim=-1),
                 'motor_return': motor_sums.sum(-1), 'discounted_motor_return': motor_discounted,
                 **{f'motor_{name}_sum': motor_sums[:, i] for i, name in enumerate(motor_names)}}
      ready = trials.update(step=t+1, roles=r, tag=tagged(env), failure=failure,
                            physics_done=physics_done, quality=quality,
                            fallen=tm('fell_over').clone(), oob=tm('robot_out').clone(),
                            failure_flags={"robot_fall": tm('fell_over').clone(),
                                           "opponent_fall": tm('opponent_fell_over').clone(),
                                           "robot_oob": wall_distance(env, 'robot') < 0,
                                           "opponent_oob": wall_distance(env, 'opponent') < 0})
      if cfg.cmd.video and not (trials.finished[0] and not ready[0]):
        frames.append(env.scene["world_cam"].data.rgb[0].cpu().numpy())
      ids = (physics_done | ready).nonzero().flatten()
      if len(ids) and not trials.finished.all():
        env.reset(env_ids=ids)
        obs = venv.get_observations()
  rows = sorted(trials.records, key=lambda r:r['env_id'])
  def near(sign):
    values = torch.tensor([r['minimum_distance_m'] for r in rows if r['role']==sign])
    return {'mean': values.mean().item(), 'p10':values.quantile(.1).item(), 'p50':values.median().item(),
            'frac_below_tag_radius':(values < g.rules.tag_radius).float().mean().item()} if len(values) else None
  speed = torch.cat(speed)
  result = {
    **summarize_trials(rows), 'trials':rows, 'post_tag_seconds':cfg.cmd.post_tag_seconds,
    'recovery_failure_definition': 'fall_only_post_tag' if cfg.cmd.post_tag_ignore_oob else 'fall_or_oob',
    'scripted_opponent_speed_scale': cfg.cmd.scripted_speed_scale if not cfg.cmd.opponent_policy else None,
    'motor_return_definition': 'own-seat regularizers through first game endpoint; excludes TAG, shaping, failure and contact-speed terms',
    'actor_observation_dim': env.observation_manager.group_obs_dim['actor'][0],
    'tag_observation_dim': tag_observation_dim(env),
    "policy": cfg.cmd.policy, "baseline": cfg.cmd.baseline, "opponent": cfg.opponent, "opponent_policy": cfg.cmd.opponent_policy,
    "mean_planar_speed": speed.mean().item(), "p95_planar_speed": speed.quantile(0.95).item(),
    "mean_abs_yaw_rate": torch.cat(yaw_rate).mean().item(),
    "min_distance_pursuer": near(1), "min_distance_evader": near(-1),
    "distance_by_second_pursuer": by_second[1.0], "distance_by_second_evader": by_second[-1.0],
    "rules": asdict(g.rules), "reset": asdict(g.reset), "env_spacing": cfg.env_spacing,
  }
  print(json.dumps(result, indent=2))
  os.makedirs(os.path.dirname(os.path.abspath(cfg.cmd.out)), exist_ok=True)
  with open(cfg.cmd.out, "w") as f:
    json.dump(result, f, indent=2)
  if cfg.cmd.video:
    import imageio_ffmpeg
    import mediapy

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(cfg.cmd.video, frames, fps=1 / env.step_dt)
    print(f"saved {cfg.cmd.video} ({len(frames)} frames)")
  env.close()


def _root(ent):
  d = ent.data
  return torch.cat([d.root_link_pos_w, d.root_link_quat_w, torch.zeros_like(d.root_link_pos_w).repeat(1, 2)], -1)


def check(cfg: DictConfig) -> None:
  """Teleport robots and assert terminations, the sparse reward, bias-free shaping and spawns."""
  g = game(cfg, shaping={"enabled": True})
  rules = g.rules
  env = make_env(cfg, cfg.cmd.num_envs, g)
  venv = OpponentEnv(env, scripted_league(env, None, cfg.device))
  zero = torch.zeros(env.num_envs, venv.num_actions, device=env.device)
  ids = torch.arange(env.num_envs, device=env.device)
  a, b = env.scene["robot"], env.scene["opponent"]
  contact, win = rules.tag_mode == "contact", rules.win_reward / 2
  near = 0.3 if contact else 0.8 * rules.tag_radius  # Bodies overlap at 0.3 m.
  assert env.max_episode_length == round(rules.episode_s / env.step_dt), env.max_episode_length
  print("game", g)

  def step_with_roles(roles):
    env.command_manager.get_term("role").role[:, 0] = torch.tensor(roles, device=env.device)
    r = role(env).clone()
    _, rew, dones, _ = venv.step(zero)
    return outcome(env, r).tolist(), rew.tolist(), dones.tolist()

  # Envs 0-1: tag (opponent inside the tag radius, or bodies overlapping in contact mode);
  # 2/3: robot/opponent outside the arena; 4: untouched, or in contact mode the opponent
  # 0.45 m behind the robot, facing its back (~0.2 m geom gap; facing each other, 0.45 m
  # touches); 5: untouched (feet on the ground only). 4-5 should keep going.
  sa, sb = _root(a), _root(b)
  ahead = (sb[:, :3] - sa[:, :3]) / (sb[:, :3] - sa[:, :3]).norm(dim=-1, keepdim=True)
  sb[:2, :3] = sa[:2, :3] + near * ahead[:2]
  if contact:
    yaw = torch.atan2(ahead[4:5, 1], ahead[4:5, 0])
    sa[4, 3:7] = sb[4, 3:7] = quat_from_euler_xyz(0 * yaw, 0 * yaw, yaw)[0]
    sb[4, :3] = sa[4, :3] - 0.45 * ahead[4]
  sa[2, 0] = env.scene.env_origins[2, 0] + rules.arena_half_extent + 0.5
  sb[3, 0] = env.scene.env_origins[3, 0] + rules.arena_half_extent + 0.5
  a.write_root_state_to_sim(sa, env_ids=ids)
  b.write_root_state_to_sim(sb, env_ids=ids)
  env.sim.forward()
  d = distance(env).tolist()
  out, rew, dones = step_with_roles([1, -1, 1, -1, 1, -1])
  print("distance", [round(x, 2) for x in d])
  print("teleport outcome", out, "reward", [round(x, 2) for x in rew], "dones", dones)
  assert out == [1, -1, 0, 0, 0, 0], out
  assert dones == [1, 1, 1, 1, 0, 0], dones
  assert rew[0] > win and rew[1] < -win and rew[2] < -win and abs(rew[3]) < win, rew

  venv.reset()  # Fresh spawns, then timeout: evader wins.
  env.episode_length_buf[:] = env.max_episode_length - 1
  out, rew, dones = step_with_roles([1, -1, 1, -1, 1, -1])
  print("timeout outcome", out, "reward", [round(x, 2) for x in rew])
  assert out == [-1, 1, -1, 1, -1, 1] and all(dones), out

  check_shaping(env, venv, g, near)
  check_spawn(cfg, g)
  check_selfplay(cfg, g, near)
  check_league(cfg, g, near)
  print("check OK")


def check_shaping(env, venv, g, near) -> None:
  """Forced episodes of several lengths and outcomes. Each step's shaping reward must be
  weight * (gamma * phi(s') * (1 - done) - phi(s)), with phi read here after each step (so
  after a reset, phi(s) is the new episode's phi(s0)), and each episode's discounted sum
  must be -weight * phi(s0)."""
  gamma, w = g.shaping.gamma, g.shaping.weight
  a, b = env.scene["robot"], env.scene["opponent"]
  ids = torch.arange(env.num_envs, device=env.device)
  term = env.reward_manager.active_terms.index("shaping")
  phi = lambda: POTENTIALS[g.shaping.potential](env)  # Read after the step: post-reset for done envs.
  ends = ("tagged", "robot_out", "opponent_out", "time_out", "fell_over", "opponent_fell_over")
  # env -> (step of the forced terminal, how): two tags, two outs, two timeouts.
  force = {0: (6, "tag"), 1: (13, "tag"), 2: (4, "out"), 3: (10, "out"), 4: (8, "time"), 5: (15, "time")}
  venv.reset()
  for e, (k, how) in force.items():
    if how == "time":
      env.episode_length_buf[e] = env.max_episode_length - k
  torch.manual_seed(0)
  prev = phi()
  phi0, ret, t = prev.clone(), torch.zeros_like(prev), torch.zeros_like(prev)
  fresh = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  err, err_fresh, episodes = 0.0, 0.0, []
  for step in range(1, 21):
    for e, (k, how) in force.items():  # Teleport just before the forced terminal step.
      if step == k and how != "time":
        sa, sb, i = _root(a)[e : e + 1], _root(b)[e : e + 1], ids[e : e + 1]
        if how == "tag":
          sb[0, :3] = sa[0, :3] + near * (sb[0, :3] - sa[0, :3]) / (sb[0, :3] - sa[0, :3]).norm()
        else:
          sa[0, 0] = env.scene.env_origins[e, 0] + g.rules.arena_half_extent + 0.5
        a.write_root_state_to_sim(sa, env_ids=i)
        b.write_root_state_to_sim(sb, env_ids=i)
    _, _, dones, _ = venv.step(0.3 * torch.randn(env.num_envs, venv.num_actions, device=env.device))
    done, now = dones.bool(), phi()  # now: phi(s') for live envs, the post-reset phi(s0) for done.
    r = env.reward_manager._step_reward[:, term] * env.step_dt  # Shaping reward actually paid.
    gap = (r - w * (gamma * now * ~done - prev)).abs()
    err, err_fresh = max(err, gap.max().item()), max(err_fresh, gap[fresh].max().item() if fresh.any() else 0)
    ret += gamma**t * r
    t += 1
    for e in done.nonzero()[:, 0].tolist():
      why = "+".join(k for k in ends if env.termination_manager.get_term(k)[e])
      episodes.append((e, int(t[e]), why, ret[e].item(), -w * phi0[e].item()))
    ret[done], t[done], phi0[done] = 0.0, 0.0, now[done]
    prev, fresh = now, done
  print(f"shaping per-step |F - formula| max {err:.2e} (first steps after reset: {err_fresh:.2e})")
  for e, n, why, total, want in episodes:
    print(f"  env {e} len {n:2d} {why:12s} sum gamma^t F = {total:+.6f}  -w*phi(s0) = {want:+.6f}")
  assert err < 1e-5 and err_fresh < 1e-5, (err, err_fresh)
  assert all(abs(total - want) < 1e-4 for *_, total, want in episodes), episodes
  whys = {why for *_, why, _, _ in episodes}
  assert {"tagged", "time_out"} <= whys and whys & {"robot_out", "opponent_out"}, whys


def check_selfplay(cfg, g, near) -> None:
  """Self-play seats: obs/action/reward shapes and seat mapping, per-seat rewards (tag
  cancels, each regularizer counted once), bias-free shaping for the opponent seat, and
  selfplay_frac=0 returning the env's own buffers untouched."""
  base = make_env(cfg, cfg.cmd.num_envs, g, eval=True)  # selfplay_frac=0: plain pass-through.
  v0 = OpponentEnv(base, scripted_league(base, None, cfg.device))
  obs, rew, dones, _ = v0.step(torch.randn(base.num_envs, v0.num_actions, device=base.device))
  assert v0.num_envs == base.num_envs and set(obs.keys()) == {"actor", "critic"}
  assert all(torch.equal(obs[k], base.obs_buf[k]) for k in ("actor", "critic")) and torch.equal(rew, base.reward_buf)
  del v0, base

  env = make_env(cfg, cfg.cmd.num_envs, g, eval=True, selfplay=True)
  venv = OpponentEnv(env, scripted_league(env, None, cfg.device), selfplay_frac=0.5)
  N, n, A = env.num_envs, venv.n_sp, venv.num_actions
  names = env.reward_manager.active_terms
  ids = torch.arange(N, device=env.device)
  a, b = env.scene["robot"], env.scene["opponent"]
  obs = venv.get_observations()
  om = env.observation_manager.compute()
  print(f"selfplay: N={N} n_sp={n} rsl-rl envs={venv.num_envs} actor {tuple(obs['actor'].shape)} critic {tuple(obs['critic'].shape)}")
  print("  reward terms:", names)
  assert venv.num_envs == N + n and obs.batch_size[0] == N + n
  for k, opp in (("actor", "opponent"), ("critic", "opponent_critic")):
    assert om[opp].shape[1] == om[k].shape[1], (k, om[opp].shape, om[k].shape)
    assert torch.equal(obs[k][:N], om[k]) and torch.equal(obs[k][N:], om[opp][:n]), k
  assert torch.equal(obs["actor"][N:, -tag_observation_dim(env) + 4], -role(env)[:n]), "opponent seat sees the negated role"
  assert set(names) >= {PREFIX + k for k in names if not k.startswith(PREFIX) and k not in ZERO_SUM}

  # Actions: seat k -> its robot; league (zero placeholder) drives non-self-play opponents.
  act = torch.zeros(N + n, A, device=env.device)
  act[N] = 1.0  # Opponent seat of env 0 only.
  q0 = [e.data.joint_pos.clone() for e in (a, b)]
  for _ in range(5):
    venv.step(act)
  raw = [env.action_manager.get_term(t).raw_action for t in ("joint_pos", "opponent")]
  assert torch.equal(raw[0], act[:N]) and torch.equal(raw[1][:n], act[N:]) and not raw[1][n:].any()
  moved = [(e.data.joint_pos - q).abs().mean(-1) for e, q in zip((a, b), q0)]
  print(f"  joint |dq| after 5 steps, env 0: robot {moved[0][0]:.4f}, opponent {moved[1][0]:.4f}; env 1 opponent {moved[1][1]:.4f}")
  assert moved[1][0] > 5 * max(moved[0][0], moved[1][1]), moved

  # Rewards: forced tag in self-play env 0 (pursuer) and env 1 (evader).
  venv.reset()
  env.command_manager.get_term("role").role[:2, 0] = torch.tensor([1.0, -1.0], device=env.device)
  sa, sb = _root(a), _root(b)
  sb[:2, :3] = sa[:2, :3] + near * (sb[:2, :3] - sa[:2, :3]) / (sb[:2, :3] - sa[:2, :3]).norm(dim=-1, keepdim=True)
  a.write_root_state_to_sim(sa, env_ids=ids)
  b.write_root_state_to_sim(sb, env_ids=ids)
  env.sim.forward()
  _, rew, dones, _ = venv.step(torch.zeros(N + n, A, device=env.device))
  r = env.reward_manager._step_reward * env.step_dt
  zs = r[:, [names.index(k) for k in ZERO_SUM if k in names]].sum(-1)
  opp = torch.tensor([k.startswith(PREFIX) for k in names], device=env.device)
  robot_seat, opp_seat = rew[:N], rew[N:]
  print(f"  tag step: robot seats {[round(x, 3) for x in robot_seat.tolist()]}, opponent seats {[round(x, 3) for x in opp_seat.tolist()]}")
  assert torch.allclose(robot_seat, r[:, ~opp].sum(-1)) and torch.allclose(opp_seat, (r[:, opp].sum(-1) - zs)[:n])
  assert torch.allclose(robot_seat[:n] + opp_seat, (env.reward_buf - zs)[:n], atol=1e-5), "each regularizer once"
  win = g.rules.win_reward / 2
  assert robot_seat[0] > win and opp_seat[0] < -win and robot_seat[1] < -win and opp_seat[1] > win, rew
  assert dones[:2].all() and torch.equal(dones[N:], dones[:n])
  assert (r[:, names.index(PREFIX + "upright")] > 0).all(), "opponent regularizers are live"

  # Opponent-seat shaping is -F_robot each step, so its discounted episode sum is
  # -w * phi_opp(s0) = +w * phi_robot(s0): bias-free for both seats.
  if "shaping" in names:
    k, gamma, w = 12, g.shaping.gamma, g.shaping.weight
    venv.reset()
    env.episode_length_buf[:] = env.max_episode_length - k  # Every episode times out at step k.
    phi0, ret, si = POTENTIALS[g.shaping.potential](env).clone(), 0.0, names.index("shaping")
    for t in range(k):
      venv.step(torch.zeros(N + n, A, device=env.device))
      ret = ret + gamma**t * -(env.reward_manager._step_reward[:, si] * env.step_dt)  # Opponent seat's share.
    print(f"  opponent-seat shaping sum {[round(x, 6) for x in ret[:n].tolist()]} vs -w*phi_opp(s0) {[round(x, 6) for x in (w * phi0[:n]).tolist()]}")
    assert torch.allclose(ret[:n], w * phi0[:n], atol=1e-4), (ret, phi0)
  print("selfplay OK")


def check_league(cfg, g, near) -> None:
  """League integration: partition bookkeeping, seeded group draws, frozen snapshots, obs and
  action routing per group, outcome attribution, and default-league == scripted-only bitwise."""
  base = LeagueCfg(sampling="uniform")
  part = League(replace(base, selfplay_frac=0.5, groups=8)).assign(4096)
  sizes = [s.stop - s.start for s in part.groups]
  print(f"league: 4096 envs, selfplay_frac 0.5, 8 groups -> {part.selfplay.stop} self-play envs ({2 * part.selfplay.stop} seats), groups {sizes}")
  assert part.selfplay == slice(0, 2048) and sizes == [256] * 8 and part.groups[0].start == 2048 and part.groups[-1].stop == 4096

  def pool(seed):
    lg = League(replace(base, groups=4), seed=seed)
    for i in range(5):
      lg.add(Opponent(f"m{i}", None, "historical"))
    lg.assign(16)
    return [[o.opponent_id for o in lg.resample_groups(it)] for it in range(6)]
  assert pool(3) == pool(3) and pool(3) != pool(4), "group draws follow the seed"

  # Default league (scripted only, no self-play, 1 group) == FixedLeague(scripted), bitwise.
  # On CPU: mujoco_warp on CUDA is not bitwise reproducible even for identical code.
  cpu = OmegaConf.merge(cfg, {"device": "cpu"})
  stage1 = load_policy(cfg.cmd.stage1, "cpu") if cfg.cmd.stage1 else None
  runs = []
  for kind in ("fixed", "league"):
    env = make_env(cpu, cfg.cmd.num_envs, g)
    scripted = Opponent(SCRIPTED_ID, ScriptedPolicy(env, stage1), SCRIPTED)
    lg = FixedLeague(scripted) if kind == "fixed" else League(LeagueCfg())
    if kind == "league":
      lg.add(scripted)
    venv, gen = OpponentEnv(env, lg), torch.Generator(device=env.device).manual_seed(1)
    rec = [venv.get_observations()["actor"].clone()]
    for _ in range(30):
      obs, rew, dones, _ = venv.step(0.5 * torch.randn(env.num_envs, venv.num_actions, device=env.device, generator=gen))
      rec += [obs["actor"].clone(), obs["critic"].clone(), rew.clone(), dones.clone()]
    runs.append(rec)
    del venv, env
  same = all(torch.equal(x, y) for x, y in zip(*runs))
  print(f"  default league vs FixedLeague(scripted, stage1={'yes' if stage1 else 'zero'}): 30 steps bitwise identical: {same}")
  assert same

  # Routing and outcomes: 6 envs = 2 self-play + groups of 2, each driven by a recording opponent.
  env = make_env(cfg, cfg.cmd.num_envs, g, eval=True, selfplay=True)
  lg = League(replace(base, selfplay_frac=1 / 3, groups=2, snapshot_every=1, resample_every=10**6), seed=0)
  A, seen = env.action_manager.get_term("opponent").action_dim, {}
  def recorder(name, value):
    def policy(o):
      seen[name] = o.clone()
      return torch.full((len(o), A), value, device=o.device)
    return policy
  for name, value in (("hist/a", 0.25), ("hist/b", -0.25)):
    lg.add(Opponent(name, recorder(name, value), "historical"))
  venv = OpponentEnv(env, lg, steps_per_iter=1)
  N, n = env.num_envs, venv.n_sp
  actor = MLPModel(venv.get_observations(), {"actor": ["actor"]}, "actor", venv.num_actions, [32], obs_normalization=True).to(env.device)
  venv.learner = actor
  obs_opp = env.obs_buf["opponent"].clone()
  act = torch.randn(N + n, venv.num_actions, device=env.device)
  venv.step(act)  # Step 0: draws; no snapshot at iteration 0.
  raw = env.action_manager.get_term("opponent").raw_action
  assert torch.equal(raw[:n], act[N:]), "self-play opponent seats act from the learner"
  for o, s in zip(venv.opponents, venv.part.groups):
    want = {"hist/a": 0.25, "hist/b": -0.25}[o.opponent_id]
    assert torch.equal(raw[s], torch.full_like(raw[s], want)), (o.opponent_id, s)
  last = {o.opponent_id: s for o, s in zip(venv.opponents, venv.part.groups)}
  for name, s in last.items():
    assert torch.equal(seen[name], obs_opp[s]), f"{name} saw its own group's opponent obs"
  print(f"  routing: self-play {venv.part.selfplay}, groups {[(o.opponent_id, s) for o, s in zip(venv.opponents, venv.part.groups)]}")

  venv.step(torch.zeros(N + n, venv.num_actions, device=env.device))  # Iteration 1: snapshot.
  snap = [o for o in lg.members if o.category == "learner"]
  probe = venv.get_observations()["actor"][:N]
  before = snap[0].policy(probe).clone()
  with torch.no_grad():
    for prm in actor.parameters():
      prm.add_(1.0)
  assert len(snap) == 1 and snap[0].opponent_id == "learner/1" and torch.equal(snap[0].policy(probe), before), "snapshot frozen"
  assert not torch.equal(actor(TensorDict({"actor": probe}, batch_size=[N])), before)

  # Outcome attribution: force a tag in the first env of group 0 with the learner as pursuer.
  o0, s0 = venv.opponents[0], venv.part.groups[0]
  games0, win0 = lg.games[o0.opponent_id], lg.win[o0.opponent_id]
  env.command_manager.get_term("role").role[s0.start, 0] = 1.0
  a, b = env.scene["robot"], env.scene["opponent"]
  sa, sb, i = _root(a)[s0.start : s0.start + 1], _root(b)[s0.start : s0.start + 1], torch.tensor([s0.start], device=env.device)
  sb[0, :3] = sa[0, :3] + near * (sb[0, :3] - sa[0, :3]) / (sb[0, :3] - sa[0, :3]).norm()
  b.write_root_state_to_sim(sb, env_ids=i)
  env.sim.forward()
  _, _, dones, extras = venv.step(torch.zeros(N + n, venv.num_actions, device=env.device))
  print(f"  outcome: {o0.opponent_id} games {games0} -> {lg.games[o0.opponent_id]}, win EMA {win0:.3f} -> {lg.win[o0.opponent_id]:.3f}; log keys {sorted(k for k in extras['log'] if k.startswith('league/win_rate'))}")
  assert dones[s0.start] and lg.games[o0.opponent_id] == games0 + 1 and lg.win[o0.opponent_id] > win0
  # Physical failures incur own penalties, but are not game wins/draws for PFSP.
  games_before = dict(lg.games)
  sa = _root(a)
  sa[:, 0] = env.scene.env_origins[:, 0] + g.rules.arena_half_extent + 0.5
  a.write_root_state_to_sim(sa, env_ids=torch.arange(N, device=env.device))
  env.sim.forward()
  _, rew, dones, _ = venv.step(torch.zeros(N+n, A, device=env.device))
  assert dones[:N].all() and (outcome(env, role(env)) == 0).all()
  assert all(lg.games[k] == count for k, count in games_before.items())
  assert (rew[:N] < -g.rules.win_reward/2).all(), 'own OOB penalty missing'
  print("league OK")


def check_spawn(cfg, g) -> None:
  """Many resets: separations in [d_min, d_max] with density peaking at d_min and falling
  monotonically, both robots >= wall_margin inside, nothing tagged or terminated at step 0."""
  env = make_env(cfg, cfg.cmd.spawn_envs, g)
  venv = OpponentEnv(env, scripted_league(env, None, cfg.device))
  zero = torch.zeros(env.num_envs, venv.num_actions, device=env.device)
  r = g.reset
  room = g.rules.arena_half_extent - r.wall_margin
  seps, xys = [], []
  for _ in range(cfg.cmd.spawn_resets):
    venv.reset()
    pa, pb = (env.scene[k].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2] for k in ("robot", "opponent"))
    seps.append((pa - pb).norm(dim=-1).cpu())
    xys.append(torch.stack([pa, pb]).cpu())
    _, _, dones, _ = venv.step(zero)
    ended = [k for k in env.termination_manager.active_terms if env.termination_manager.get_term(k).any()]
    assert not dones.any(), ended
  d, xy = torch.cat(seps), torch.cat(xys, 1)
  inside = xy.abs().max().item()
  print(f"spawn: {len(d)} pairs, d in [{d.min():.4f}, {d.max():.4f}], max |xy| {inside:.4f} <= {room}")
  assert r.d_min - 1e-4 <= d.min() and d.max() <= r.d_max + 1e-4 and inside <= room + 1e-4
  if r.d_max > r.d_min:
    edges = torch.linspace(r.d_min, r.d_max, 11)
    counts = torch.histc(d, bins=10, min=r.d_min, max=r.d_max).long()
    for lo, hi, c in zip(edges[:-1].tolist(), edges[1:].tolist(), counts.tolist()):
      print(f"  d [{lo:.3f}, {hi:.3f}) {c:6d} {'#' * round(60 * c / counts.max().item())}")
    assert (counts[1:] < counts[:-1]).all(), counts  # Mode in the first bin, then decreasing.
  if cfg.cmd.plots:
    plot_spawn(d, xy[:, : cfg.cmd.spawn_envs], g, cfg.cmd.plots)  # Scatter: the first reset.


def plot_spawn(d, xy, g, out_dir) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  r, half = g.reset, g.rules.arena_half_extent
  fig, (h, s) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]})
  h.hist(d.numpy(), bins=34, range=(r.d_min, r.d_max), color="#2a78d6", edgecolor="white", linewidth=1)
  h.set(xlabel="separation d (m)", ylabel="spawns", title=f"Spawn separation: {len(d)} spawns, {r.separation}")
  h.spines[["top", "right"]].set_visible(False)
  for i, (name, color) in enumerate((("robot", "#2a78d6"), ("opponent", "#eb6834"))):
    s.scatter(*xy[i].T.numpy(), s=3, alpha=0.4, color=color, label=name, linewidths=0)
  for lim, style in ((half, "-"), (half - r.wall_margin, "--")):
    s.plot([-lim, lim, lim, -lim, -lim], [-lim, -lim, lim, lim, -lim], style, color="#555", linewidth=1)
  s.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Spawn positions, first reset (arena, wall margin)")
  s.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, markerscale=4, frameon=False)
  fig.tight_layout()
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f"spawn_{r.placement}_{r.separation}.png")
  fig.savefig(path, dpi=120)
  print(f"saved {path}")


def main(cfg: DictConfig) -> None:
  cfg.device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  {"tag": train, "eval": evaluate, "check": check}[HydraConfig.get().runtime.choices.cmd](cfg)


if __name__ == "__main__":
  run(main, "mofo_train", SCHEMAS, commands=("tag", "eval", "check"))
