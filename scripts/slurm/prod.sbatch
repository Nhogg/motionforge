#!/bin/bash
# Run any command on a production GPU node; "{job}" in args becomes the job id.
# Submit from the repo root; the job runs in the submit directory. Override resources with sbatch flags, e.g.:
#   sbatch scripts/slurm/prod.sbatch uv run scripts/training/train_hierarchical_tag.py seed=0 output_dir='logs/hierarchical/training/run-{job}'
#   sbatch --nodelist=g4 --array=0-7%8 --time=06:00:00 scripts/slurm/prod.sbatch bash sweep.sh
#SBATCH --job-name=prod-motion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.log
set -euo pipefail
export MUJOCO_GL=disable  # Render with Warp; never load EGL.
export UV_NO_SYNC=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
hostname; nvidia-smi -L
time "${@//\{job\}/$SLURM_JOB_ID}"
