"""wandb.py.

Author: Nathan Hogg <nathanhogg1223@gmail.com>
Description:
    Weights & Biases metrics logging.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

WandbMode = Literal["disabled", "online", "offline"]


class WandbMetricsLogger:
    """Mirror experiment configuration and scalar metrics to W&B."""

    def __init__(
        self,
        *,
        mode: WandbMode,
        project: str,
        entity: str | None,
        name: str,
        group: str | None,
        output_dir: Path,
        configuration: Mapping[str, Any],
    ) -> None:
        self._mode = mode
        self._run = None

        if mode == "disabled":
            return

        import wandb

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type="training",
            mode=mode,
            dir=output_dir,
            config=dict(configuration),
        )
        self._run.define_metric("environment_steps")
        self._run.define_metric(
            "*",
            step_metric="environment_steps",
        )

    @property
    def metadata(self) -> dict[str, Any]:
        if self._run is None:
            return {
                "enabled": False,
                "mode": self._mode,
            }

        return {
            "enabled": True,
            "entity": self._run.entity,
            "group": self._run.group,
            "id": self._run.id,
            "mode": self._mode,
            "name": self._run.name,
            "project": self._run.project,
            "url": self._run.url,
        }

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log_video(
        self,
        *,
        environment_steps: int,
        path: Path,
        caption: str,
        key: str = "evaluation/policy_video",
    ) -> None:
        if self._run is None:
            return

        import wandb

        self._run.log(
            {
                "environment_steps": environment_steps,
                key: wandb.Video(
                    str(path),
                    caption=caption,
                    format="mp4",
                ),
            }
        )

    def log(
        self,
        *,
        environment_steps: int,
        metrics: Mapping[str, Any],
    ) -> None:
        if self._run is None:
            return

        self._run.log(
            {
                "environment_steps": environment_steps,
                **metrics,
            }
        )

    def finish(
        self,
        summary: Mapping[str, Any],
    ) -> None:
        if self._run is None:
            return

        self._run.summary.update(dict(summary))
        self._run.finish()
