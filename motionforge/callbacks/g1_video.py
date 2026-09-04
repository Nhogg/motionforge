"""g1_video.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Scheduled G1 policy-video callback.

    The callback renders at step zero and then at fixed environment-step
    interval. Every video and JSON sidecar is tained locally. When a W&B
    logger is supplied, the MP4 is also attached to the active run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from motionforge.evaluation.g1_video import render_g1_policy_video
from motionforge.logging.wandb import WandbMetricsLogger


@dataclass
class G1VideoCallback:
    environment: Any
    logger: WandbMetricsLogger
    output_dir: Path

    interval_steps: int = 20_000_000
    duration: float = 5.0
    fps: int = 15
    width: int = 640
    height: int = 480
    seed: int = 0
    command: tuple[float, float, float] = (0.0, 0.0, -0.5)
    render_step_zero: bool = True

    _last_rendered_step: int | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.interval_steps <= 0:
            raise ValueError("Video callback interval must be positive")

    def should_render(self, step: int) -> bool:
        if self._last_rendered_step is None:
            return self.render_step_zero or step > 0

        return step - self._last_rendered_step >= self.interval_steps

    def __call__(
        self,
        current_step: int,
        make_policy: Callable[..., Any],
        parameters: Any,
    ) -> None:
        step = int(current_step)

        if not self.should_render(step):
            return

        video_dir = self.output_dir / "videos"
        video_path = video_dir / f"policy_{step:012d}.mp4"
        metadata_path = video_path.with_suffix(".json")

        metadata = render_g1_policy_video(
            environment=self.environment,
            make_policy=make_policy,
            parameters=parameters,
            output=video_path,
            seed=self.seed,
            command=self.command,
            duration=self.duration,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        metadata["environment_steps"] = step

        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        self.logger.log_video(
            environment_steps=step,
            path=video_path,
            caption=(f"step={step}, command={list(self.command)}, seed={self.seed}"),
        )
        self._last_rendered_step = step
