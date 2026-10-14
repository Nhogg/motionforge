#!/usr/bin/env bash
#SBATCH --job-name=tag-path-audit
#SBATCH --partition=gpu
#SBATCH --nodelist=g3
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
cd /home/nhogg/motionforge
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
ARGS=()
ROLE=pursuer
case "${TASK_ID}" in
  0|1|6)
    CHECKPOINT=logs/hierarchical/training/tag_clean_start_mixed_4096_seed3_g4/checkpoints/000000001250
    SEED=3
    CASE=late_failure
    ;;
  2|3|4|5|7)
    CHECKPOINT=logs/hierarchical/training/tag_pursuit_diag_mixed_frozen_seed1_job4199/checkpoints/000000000100
    SEED=1
    CASE=successful_seed_selfplay
    ;;
  *) echo "Unsupported task ${TASK_ID}" >&2; exit 2 ;;
esac
if [[ "${TASK_ID}" == 4 || "${TASK_ID}" == 5 ]]; then
  ARGS=(use_self_play=false use_training_reset=false initial_separation=2.0
    opponent_checkpoint=logs/hierarchical/training/tag_sparse_terminal_640k_seed0/checkpoints/000000000005)
  SEED=1000
  CASE=successful_seed_vs_historical005
fi
case "${TASK_ID}" in
  1|3|5|6|7) STOCHASTIC=true ;;
  *) STOCHASTIC=false ;;
esac
if [[ "${TASK_ID}" == 6 || "${TASK_ID}" == 7 ]]; then ROLE=evader; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
OUTPUT="logs/hierarchical/trajectories/role_audit_job${SLURM_ARRAY_JOB_ID}/${CASE}_${ROLE}_stochastic_${STOCHASTIC}.png"
/home/nhogg/motionforge/.venv/bin/python scripts/evaluation/plot_hierarchical_tag_trajectory.py \
  checkpoint="${CHECKPOINT}" role="${ROLE}" seed="${SEED}" \
  stochastic_actions="${STOCHASTIC}" duration=10.0 \
  "${ARGS[@]}" output="${OUTPUT}"
