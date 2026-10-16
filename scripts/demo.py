"""Smoke test: build the two-G1 env, step it with zero actions (or a stage-1 policy), check sanity.

Hydra config: config/demo.yaml.

  uv run scripts/demo.py [device=cuda:0] [steps=500] [video=OUT.mp4] [no_fall=true] [policy=STAGE1.pt]
"""

import os

os.environ.setdefault("MUJOCO_GL", "disable")  # Before mjlab import (it defaults to egl).

import hydra
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.camera_sensor import CameraSensorCfg
from omegaconf import DictConfig

from mofo.opponent import load_policy
from mofo.tasks import two_g1_env_cfg

# Fixed world camera 7 m to the side of env 0's origin, tilted ~11 deg down, framing
# both robots. Warp-rendered (mujoco_warp), so no EGL is needed.
WORLD_CAM = CameraSensorCfg(
  name="world_cam", pos=(0.0, -7.0, 2.0), quat=(0.7731, 0.6342, 0.0, 0.0), width=640, height=480
)


def base_pos(env):
  return {n: env.scene[n].data.root_link_pos_w.clone() for n in env.cfg.actions}


def main(cfg: DictConfig):
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # One env when recording: the camera is placed in world coords, and env 0's grid
  # origin is (0, 0, 0) only when it is the sole env.
  env_cfg = two_g1_env_cfg(num_envs=1 if cfg.video else 2)
  if cfg.no_fall:
    env_cfg.terminations = {k: v for k, v in env_cfg.terminations.items() if "fell_over" not in k}
  if cfg.video:
    env_cfg.scene.sensors += (WORLD_CAM,)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  print("device:", env.device)
  obs, _ = env.reset()
  frames = []
  if cfg.video:
    assert not env.scene.env_origins[0].any(), "camera assumes env 0 at the world origin"
    frames.append(env.scene["world_cam"].data.rgb[0].cpu().numpy())
  am = env.action_manager
  print("obs groups:", {k: tuple(v.shape) for k, v in obs.items()})
  print("action dim:", am.total_action_dim, dict(zip(am.active_terms, am.action_term_dim)))

  start = base_pos(env)
  # Zero actions = hold the default (knees-bent) pose, since use_default_offset=True.
  action = torch.zeros(env.num_envs, am.total_action_dim, device=env.device)
  # Stage-1 policy input is each robot's obs group followed by its velocity command.
  policy = load_policy(cfg.policy, device) if cfg.policy else None
  cmd = {n: torch.tensor(cfg.commands[n], device=env.device).expand(env.num_envs, 3) for n in am.active_terms}
  min_z = {n: p[:, 2].clone() for n, p in start.items()}
  min_gap, max_speed, min_body_z = float("inf"), 0.0, float("inf")
  tm = env.termination_manager
  for t in range(cfg.steps):
    if policy:
      with torch.inference_mode():
        action = torch.cat([policy(torch.cat([obs[n], cmd[n]], -1)) for n in am.active_terms], -1)
    obs, rew, term, trunc, _ = env.step(action)
    for name in tm.active_terms:
      if (fired := tm.get_term(name).nonzero().flatten().tolist()):
        print(f"step {t}: {name} fired in envs {fired} (whole env reset)")
    pos = base_pos(env)
    for n, p in pos.items():
      min_z[n] = torch.minimum(min_z[n], p[:, 2])
      assert torch.isfinite(p).all(), f"NaN in {n} base pos"
      data = env.scene[n].data
      max_speed = max(max_speed, data.root_link_lin_vel_w.norm(dim=-1).max().item())
      min_body_z = min(min_body_z, data.body_link_pos_w[..., 2].min().item())
    min_gap = min(min_gap, (pos["g1_a"] - pos["g1_b"]).norm(dim=-1).min().item())
    assert all(torch.isfinite(v).all() for v in obs.values()), "NaN in obs"
    assert torch.isfinite(rew).all(), "NaN in reward"
    if cfg.video:
      frames.append(env.scene["world_cam"].data.rgb[0].cpu().numpy())
  end = base_pos(env)

  for n in start:
    print(f"{n}: start {start[n].tolist()}\n      end   {end[n].tolist()}\n      min z {min_z[n].tolist()}")
  print(f"max base speed {max_speed:.2f} m/s, min base gap {min_gap:.2f} m, min body z {min_body_z:.3f} m")

  a, b = start["g1_a"], start["g1_b"]
  assert (a - b).norm(dim=-1).min() > 1.0, "robots overlap at start"
  assert min_gap > 0.3, "robots collapsed together"
  assert max_speed < 10.0, "base speed blew up"
  assert min_body_z > -0.05, "bodies sank through the floor"
  for n, p in start.items():
    assert ((p[:, 2] > 0.6) & (p[:, 2] < 0.9)).all(), f"{n} start height {p[:, 2]}"

  if cfg.video:
    import imageio_ffmpeg
    import mediapy

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    os.makedirs(os.path.dirname(os.path.abspath(cfg.video)), exist_ok=True)
    mediapy.write_video(cfg.video, frames, fps=1 / env.step_dt)
    print(f"saved {cfg.video}: {len(frames)} frames {frames[0].shape} @ {1 / env.step_dt:.0f} fps")
  print("OK")


@hydra.main(version_base="1.3", config_path="../config", config_name="demo")
def cli(cfg: DictConfig) -> None:
  main(hydra.utils.instantiate(cfg))


if __name__ == "__main__":
  cli()
