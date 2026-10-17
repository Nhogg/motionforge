"""Compatibility adapter for rsl-rl's writer and current W&B releases."""

import os

import wandb
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.utils.wandb_log_writer import WandbLogWriter


class WandbWriter(WandbLogWriter):
  def __init__(self, log_dir: str, project_name: str):
    # Upstream passes the removed Settings(start_method=...) option. Reuse its
    # scalar/config/artifact methods, but initialize without that obsolete option.
    SummaryWriter.__init__(self, log_dir, flush_secs=10)
    wandb.init(project=project_name, entity=os.environ.get("WANDB_USERNAME"),
               name=os.path.basename(os.path.normpath(log_dir)), dir=log_dir,
               config={"log_dir": log_dir})
    self.logged_videos: set[str] = set()

  def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
    # rsl-rl recognizes only the literal upstream class name when skipping its
    # wall-time duplicate curves. Keep those local so seconds cannot advance the
    # W&B iteration counter and cause later iteration metrics to be dropped.
    if tag.endswith("/time"):
      return SummaryWriter.add_scalar(self, tag, scalar_value, global_step, **kwargs)
    return super().add_scalar(tag, scalar_value, global_step, **kwargs)

  def stop(self):
    try:
      super().stop()
    finally:
      self.close()
