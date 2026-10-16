#!/usr/bin/env bash
#SBATCH --job-name=tag-baseline16
#SBATCH --partition=gpu
#SBATCH --nodelist=g4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-15%8
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

cd /home/nhogg/motionforge
mkdir -p logs/slurm logs/hierarchical/training

SEED="${SLURM_ARRAY_TASK_ID}"
HBASE=logs/hierarchical/training/tag_sparse_terminal_640k_seed0/checkpoints
HISTORICAL="[$HBASE/000000000005,$HBASE/000000000010,$HBASE/000000000015,$HBASE/000000000020]"
RUN="tag_clean_start_mixed_4096_seed${SEED}_g4"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} seed=${SEED} run=${RUN}"

uv run scripts/training/train_hierarchical_tag.py \
  locomotion_checkpoint=logs/p2/training/g1_rebuilt_full_seed0/checkpoints/000200048640 \
  "historical_opponent_checkpoints=${HISTORICAL}" \
  historical_opponent_probability=0.75 \
  minimum_initial_separation=1.5 \
  maximum_initial_separation=3.0 \
  seed="${SEED}" \
  num_environments=4096 \
  minibatch_size=1024 \
  num_iterations=10000 \
  rollout_horizon=500 \
  episode_duration=10.0 \
  checkpoint_interval=50 \
  naconmax_per_environment=64 \
  njmax=4096 \
  wandb_mode=online \
  wandb_group=hierarchical_tag_large_scale \
  wandb_name="${RUN}" \
  wandb_videos=true \
  wandb_video_interval_steps=102400000 \
  wandb_video_duration=10.0 \
  elo_baseline_pursuer_win_probability=0.65 \
  output_dir="logs/hierarchical/training/${RUN}"
