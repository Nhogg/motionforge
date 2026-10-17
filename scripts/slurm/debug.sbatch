#!/bin/bash
# Run any command on a debug GPU node; "{job}" in args becomes the job id.
# Submit from the repo root; the job runs in the submit directory, e.g.:
#   sbatch scripts/slurm/debug.sbatch uv run scripts/demo.py video='logs/slurm/video-{job}/clip.mp4'
#   sbatch --nodelist=g5 --time=00:30:00 scripts/slurm/debug.sbatch bash some_script.sh
#SBATCH --job-name=debug-motion
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x-%j.log
set -euo pipefail
# Direct invocation submits; Slurm execution runs the supplied command.
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  [[ $# -gt 0 ]] || { echo 'Supply a command to run.' >&2; exit 2; }
  mkdir -p logs/slurm
  if [[ $1 == uv ]]; then shift; set -- "$(command -v uv)" "$@"; fi
  exec sbatch "${BASH_SOURCE[0]}" "$@"
fi
export MUJOCO_GL=disable  # Render with Warp; never load EGL.
export UV_NO_SYNC=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
hostname; nvidia-smi -L
time "${@//\{job\}/$SLURM_JOB_ID}"
