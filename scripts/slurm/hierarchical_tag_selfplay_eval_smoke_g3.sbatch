#!/usr/bin/env bash
#SBATCH --job-name=tag-selfplay-eval-smoke
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
/home/nhogg/motionforge/.venv/bin/python scripts/evaluation/evaluate_hierarchical_tag.py \
  checkpoint=logs/hierarchical/training/tag_pursuit_diag_mixed_frozen_seed1_job4199/checkpoints/000000000100 \
  use_self_play=true initial_separation=1.5 episode_duration=10.0 \
  seed=2000 seeds=1 duration=0.2 naconmax=256 njmax=4096 \
  stochastic_actions=true record_trajectories=false \
  output="logs/hierarchical/evaluation/selfplay_eval_smoke_job${SLURM_JOB_ID}.json"
