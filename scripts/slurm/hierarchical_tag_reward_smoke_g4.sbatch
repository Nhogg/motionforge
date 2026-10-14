#!/usr/bin/env bash
#SBATCH --job-name=tag-reward-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=g4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /home/nhogg/motionforge
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
PYTHON=/home/nhogg/motionforge/.venv/bin/python
# Ensure shared-filesystem workers do not reuse pre-refactor bytecode.
"${PYTHON}" -m compileall -q -f motionforge scripts/tests scripts/training/train_hierarchical_tag.py scripts/evaluation/evaluate_hierarchical_tag.py
"${PYTHON}" scripts/tests/test_tag_reward_truncation_gpu.py
"${PYTHON}" scripts/tests/test_hierarchical_rollout_batch.py output="logs/hierarchical/reward_refactor_smoke_job${SLURM_JOB_ID}/batch.json"
"${PYTHON}" scripts/tests/test_hierarchical_training_smoke.py output="logs/hierarchical/reward_refactor_smoke_job${SLURM_JOB_ID}/training.json"
"${PYTHON}" scripts/tests/test_hierarchical_batched_training.py output="logs/hierarchical/reward_refactor_smoke_job${SLURM_JOB_ID}/batched.json"
"${PYTHON}" scripts/tests/test_hierarchical_video_callback.py
"${PYTHON}" scripts/training/train_hierarchical_tag.py \
  num_environments=2 minibatch_size=2 num_iterations=1 rollout_horizon=5 episode_duration=0.1 \
  freeze_locomotion=true naconmax_per_environment=64 njmax=4096 \
  wandb_mode=disabled wandb_videos=false \
  output_dir="logs/hierarchical/training/tag_reward_refactor_smoke_job${SLURM_JOB_ID}"
"${PYTHON}" scripts/evaluation/evaluate_hierarchical_tag.py \
  checkpoint="logs/hierarchical/training/tag_reward_refactor_smoke_job${SLURM_JOB_ID}/checkpoints/000000000001" \
  use_self_play=true seed=3000 seeds=1 duration=0.1 episode_duration=0.1 \
  stochastic_actions=true record_trajectories=false njmax=4096 \
  output="logs/hierarchical/reward_refactor_smoke_job${SLURM_JOB_ID}/evaluation.json"
