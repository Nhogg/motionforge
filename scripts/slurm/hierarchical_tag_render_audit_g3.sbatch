#!/usr/bin/env bash
#SBATCH --job-name=tag-render-audit
#SBATCH --partition=gpu
#SBATCH --nodelist=g3
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /home/nhogg/motionforge
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
/home/nhogg/motionforge/.venv/bin/python scripts/tests/test_hierarchical_video_callback.py
/home/nhogg/motionforge/.venv/bin/python scripts/evaluation/render_hierarchical_tag.py \
  checkpoint=logs/hierarchical/training/tag_clean_start_mixed_4096_seed3_g4/checkpoints/000000001250 \
  use_self_play=true use_training_reset=true stochastic_actions=true seed=3 \
  role=pursuer duration=10.0 fps=20 width=640 height=480 njmax=4096 \
  output="logs/hierarchical/trajectories/role_render_job${SLURM_JOB_ID}/late_failure_labeled.mp4"
