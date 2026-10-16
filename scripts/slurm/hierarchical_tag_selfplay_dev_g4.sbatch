#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/slurm/hierarchical_tag_selfplay_dev_g4.sbatch
# Run the second development seed with:
#   sbatch --export=ALL,TAG_SEED=1 scripts/slurm/hierarchical_tag_selfplay_dev_g4.sbatch
#SBATCH --job-name=tag-selfplay-dev
#SBATCH --partition=gpu
#SBATCH --nodelist=g4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd /home/nhogg/motionforge
mkdir -p logs/slurm logs/hierarchical/training

SEED="${TAG_SEED:-0}"
RUN="tag_rebuilt_p2_selfplay_dev_s${SEED}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} seed=${SEED} run=${RUN}"

.venv/bin/python scripts/training/train_hierarchical_tag.py \
  locomotion_checkpoint=logs/p2/training/g1_rebuilt_full_seed0/checkpoints/000200048640 \
  opponent_mode=self_play \
  seed="${SEED}" \
  num_environments=1024 \
  num_iterations=100 \
  rollout_horizon=500 \
  episode_duration=10.0 \
  minimum_initial_separation=1.5 \
  maximum_initial_separation=1.5 \
  initial_heading_noise_radians=0.0 \
  freeze_locomotion=true \
  strategy_gae_lambda=1.0 \
  save_initial_checkpoint=true \
  checkpoint_interval=50 \
  naconmax_per_environment=64 \
  naconmax=65536 \
  njmax=4096 \
  minibatch_size=1024 \
  wandb_mode=online \
  wandb_group=hierarchical_tag_selfplay_dev \
  wandb_name="${RUN}" \
  wandb_videos=true \
  wandb_video_interval_steps=12800000 \
  wandb_video_duration=10.0 \
  wandb_video_fps=20 \
  output_dir="logs/hierarchical/training/${RUN}"
