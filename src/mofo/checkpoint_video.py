"""Checkpoint-triggered rendering through the existing evaluation CLI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import torch

from mjlab.rl import MjlabOnPolicyRunner
from omegaconf import OmegaConf

from mofo.league import League, Opponent
from mofo.role_model import RoleModel

ROOT = Path(__file__).resolve().parents[2]


def evaluation_config(training, checkpoint):
  """Copy training rules unchanged; use deterministic first-episode evaluation."""
  cfg = OmegaConf.create(training)
  render = cfg.cmd.render
  if render.matchup not in ("selfplay", "scripted"):
    raise ValueError("cmd.render.matchup must be selfplay or scripted")
  learner_role = render.get("learner_role", "pursuer")
  if learner_role not in ("pursuer", "evader"):
    raise ValueError("cmd.render.learner_role must be pursuer or evader")
  if render.timeout_s <= 0 or render.post_tag_seconds < 0:
    raise ValueError("Render timeout must be positive and post_tag_seconds nonnegative")
  checkpoint = Path(checkpoint).resolve()
  output = checkpoint.parent / "videos" / checkpoint.stem
  reference = cfg.opponent or cfg.cmd.init
  if render.matchup == "scripted" and not reference:
    raise ValueError("Scripted checkpoint renders require opponent or cmd.init")
  cfg.cmd = OmegaConf.load(ROOT / "config/cmd/eval.yaml")
  cfg.cmd.policy = str(checkpoint)
  cfg.cmd.opponent_policy = str(checkpoint) if render.matchup == "selfplay" else None
  cfg.opponent = reference
  cfg.cmd.video, cfg.cmd.out = str(output.with_suffix(".mp4")), str(output.with_suffix(".json"))
  cfg.cmd.post_tag_seconds = render.post_tag_seconds
  cfg.num_envs, cfg.seed = 1, render.seed
  cfg.rules.learner_role = learner_role # Red pursuer, blue evader; no random role swapping.
  cfg.defaults = ["mofo_train", {"override cmd": "eval"}, "_self_"]
  cfg.hydra = {"searchpath": [f"file://{ROOT / 'config'}"],
               "run": {"dir": str(output.parent / "hydra" / checkpoint.stem)}, "job": {"chdir": False}}
  return cfg


class CheckpointVideoRunner(MjlabOnPolicyRunner):
  """Pause training while a separate process renders, preserving training RNG/state."""

  def __init__(self, *args, **kwargs):
    self._rendered = set()
    super().__init__(*args, **kwargs)

  def save(self, path, infos=None):
    infos = {**(infos or {}), 'mofo_completed_iterations': self.env.steps // self.cfg['num_steps_per_env']}
    if isinstance(self.env.league, League):
      infos['mofo_league'] = self.env.league.state_dict()
    super().save(path, infos)
    training = self.cfg["motionforge"]
    options = training["cmd"]["render"]
    if not options["enabled"] or not Path(path).stem.startswith("model_") or path in self._rendered:
      return # initial.pt is saved before the logger starts; final save can be a duplicate.
    cfg = evaluation_config(training, path)
    video, result = Path(cfg.cmd.video), Path(cfg.cmd.out)
    video.parent.mkdir(parents=True, exist_ok=True)
    config_path = video.with_suffix(".yaml")
    OmegaConf.save(cfg, config_path)
    command = [sys.executable, str(ROOT / "scripts/mofo_train.py"),
               f"--config-path={video.parent}", f"--config-name={config_path.stem}"]
    child_env = {**os.environ, "WANDB_MODE": "disabled"} # Only parent owns the W&B run.
    print(f"[checkpoint render] {path} -> {video}", flush=True)
    try:
      with video.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=options["timeout_s"])
      if not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError(f"Renderer did not produce {video}")
      with result.open() as f:
        summary = json.load(f)
      writer = self.logger.writer
      if writer is not None:
        writer.add_scalar("Render/success", 1.0, self.current_learning_iteration)
        if hasattr(writer, "save_video"):
          writer.save_video(video, self.current_learning_iteration)
        for key in ("tag_rate_pursuer", "controlled_tag_rate_pursuer", "physical_failure_rate_pursuer",
                    "win_rate_evader", "physical_failure_rate_evader"):
          value = summary.get(key)
          if isinstance(value, (int, float)):
            writer.add_scalar(f"Render/{key}", value, self.current_learning_iteration)
      self._rendered.add(path)
    except Exception as error:
      if self.logger.writer is not None:
        self.logger.writer.add_scalar("Render/success", 0.0, self.current_learning_iteration)
      if options["fail_on_error"]:
        raise
      warnings.warn(f"Checkpoint render failed: {error}; see {video.with_suffix('.log')}", stacklevel=2)

  def resume(self, path, restore_league=True, restore_optimizer=True, evader_path=None):
    """Resume model/optimizer/clock; physics resets. Legacy league EMA/RNG were not saved.

    ``evader_path``: a second shared TAG checkpoint for the evader branch when migrating
    to separate roles; ``path`` then initializes the pursuer branch, clock and league.
    """
    saved = torch.load(path, map_location=self.device, weights_only=False)
    migrating = isinstance(self.alg.actor, RoleModel) and "_role_column" not in saved['actor_state_dict']
    if evader_path and not migrating:
      raise ValueError('cmd.resume_evader requires agent.separate_roles=true and a shared cmd.resume checkpoint')
    if migrating:
      if restore_optimizer:
        raise ValueError('Shared-to-separate roles requires explicit cmd.resume_optimizer=false')
      for group in ('actor', 'critic'):
        getattr(self.alg, group).load_shared_state_dict(saved[f'{group}_state_dict'])
      if evader_path:
        evader = torch.load(evader_path, map_location=self.device, weights_only=False)
        if "_role_column" in evader['actor_state_dict']:
          raise ValueError('cmd.resume_evader must be a shared-policy checkpoint')
        for group in ('actor', 'critic'):
          getattr(self.alg, group).evader.load_state_dict(evader[f'{group}_state_dict'])
        print(f'[resume] pursuer branch from {path}; evader branch from {evader_path}')
      self.current_learning_iteration = saved['iter']
      infos = saved.get('infos') or {}
    else:
      infos = super().load(path, map_location=self.device,
                          load_cfg={'actor': True, 'critic': True, 'optimizer': restore_optimizer,
                                    'iteration': True}) or {}
    print(f'[resume] optimizer {"restored" if restore_optimizer else "fresh"}; separate-role migration={migrating}')
    completed = infos.get('mofo_completed_iterations')
    if completed is None:
      completed = 0 if Path(path).stem == 'initial' else self.current_learning_iteration + 1
    self.current_learning_iteration = int(completed)
    self.env.steps = int(completed) * self.cfg['num_steps_per_env']
    league = self.env.league
    if restore_league and isinstance(league, League):
      if 'mofo_league' in infos:
        league.load_state_dict(infos['mofo_league'])
        print('[resume] restored saved league members, EMA and sampler RNG')
      else:
        snapshots = Path(path).resolve().parent.joinpath('league').glob('learner_*.pt')
        for snapshot in sorted(snapshots, key=lambda p: int(p.stem.split('_')[-1])):
          iteration = int(snapshot.stem.split('_')[-1])
          name = f'learner/{iteration}'
          if iteration < completed and name not in league.win:
            league.add(Opponent(name, league.load(str(snapshot)), 'learner', str(snapshot)))
        print('[resume] legacy league rebuilt from available snapshots; EMA and sampler RNG restart')
    elif not restore_league:
      print('[resume] using the explicitly configured new opponent pool')
    print(f'[resume] actor/critic restored; next iteration {completed}, physics reset, league size {len(league.members)}')
