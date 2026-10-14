"""Publish existing role-audit plots and the corrected video to a diagnostic W&B run."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from motionforge.logging.wandb import WandbMetricsLogger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sources = [
        root / f"logs/hierarchical/trajectories/role_audit_job{job}"
        for job in (4257, 4265)
    ]
    images = [path for directory in sources for path in directory.glob("*.png")]
    video = (
        root
        / "logs/hierarchical/trajectories/role_render_job4268/late_failure_labeled.mp4"
    )
    if len(images) != 8 or not video.is_file():
        raise FileNotFoundError(
            "expected eight generated trajectory plots and corrected video"
        )
    source_manifest = (
        root
        / "logs/hierarchical/training/tag_pursuit_diag_mixed_frozen_seed1_job4199/manifest.json"
    )
    source_run = json.loads(source_manifest.read_text())["wandb"]
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output = root / f"logs/hierarchical/diagnostics/renderer_audit_wandb_{stamp}"
    output.mkdir(parents=True)
    logger = WandbMetricsLogger(
        mode=args.mode,
        project=source_run["project"],
        entity=source_run["entity"],
        name=f"tag_renderer_trajectory_audit_{stamp}",
        group="hierarchical_tag_renderer_audit",
        output_dir=output,
        configuration={
            "purpose": "checkpoint trajectory and renderer audit",
            "source_run_url": source_run["url"],
            "plot_count": len(images),
            "colors": {"green": "pursuer", "orange": "evader"},
            "source_directories": [str(path) for path in sources],
        },
    )
    for path in sorted(images):
        metadata = json.loads(path.with_suffix(".json").read_text())
        logger.log_image(
            environment_steps=metadata["checkpoint_environment_steps"],
            path=path,
            caption=f"{metadata['opponent_id']}; seed={metadata['seed']}; stochastic={metadata['stochastic_actions']}; roles validated",
            key=f"audit/{path.stem}/trajectory",
        )
    logger.log_video(
        environment_steps=2_560_000_000,
        path=video,
        caption="Correctly labeled self-play: green=pursuer, orange=evader; seed3 iteration1250",
        key="audit/late_failure_labeled_video",
    )
    logger.finish(
        {
            "plots_uploaded": len(images),
            "video_uploaded": True,
            "role_routing_verified": True,
            "interpretation": "late pursuit failure is real; historical-opponent success does not establish self-play success",
        }
    )
    (output / "published.json").write_text(json.dumps(logger.metadata, indent=2) + "\n")
    print(json.dumps(logger.metadata))


if __name__ == "__main__":
    main()
