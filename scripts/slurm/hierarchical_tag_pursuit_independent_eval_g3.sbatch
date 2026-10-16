#!/usr/bin/env bash
#SBATCH --job-name=tag-pursuit-transfer
#SBATCH --partition=gpu
#SBATCH --nodelist=g3
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
cd /home/nhogg/motionforge
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
if (( TASK_ID < 0 || TASK_ID > 7 )); then
  echo "Unsupported task ${TASK_ID}" >&2
  exit 2
fi
LEARNER_INDEX="$((TASK_ID / 4))"
OPPONENT_INDEX="$(((TASK_ID % 4) / 2))"
if (( TASK_ID % 2 == 0 )); then STOCHASTIC=false; else STOCHASTIC=true; fi
if (( LEARNER_INDEX == 0 )); then VARIANT=mixed_frozen; else VARIANT=mixed_joint; fi
LEARNER="logs/hierarchical/training/tag_pursuit_diag_${VARIANT}_seed1_job4199/checkpoints/000000000100"
if (( OPPONENT_INDEX == 0 )); then
  OPPONENT=logs/hierarchical/training/tag_clean_start_mixed_4096_seed4_g4/checkpoints/000000000950
  OPPONENT_NAME=baseline_seed4_iter950
else
  OPPONENT=logs/hierarchical/training/tag_ablation_historical_050_strategy_lr_1e3_seed2_job4004/checkpoints/000000000500
  OPPONENT_NAME=ablation_seed2_iter500
fi
LOCOMOTION=logs/p2/training/g1_rebuilt_full_seed0/checkpoints/000200048640
for CHECKPOINT in "${LOCOMOTION}" "${LEARNER}" "${OPPONENT}"; do
  if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
  fi
done

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
OUTPUT="logs/hierarchical/evaluation/pursuit_independent_job${SLURM_ARRAY_JOB_ID}/${VARIANT}_vs_${OPPONENT_NAME}_stochastic_${STOCHASTIC}.json"
mkdir -p logs/slurm
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} learner=${LEARNER} opponent=${OPPONENT} stochastic=${STOCHASTIC}"

# These opponents were absent from array4199's training league. This remains
# the existing symmetric face-to-face 2 m reset protocol, not a wider reset test.
uv run scripts/evaluation/evaluate_hierarchical_tag.py \
  checkpoint="${LEARNER}" opponent_checkpoint="${OPPONENT}" \
  locomotion_checkpoint="${LOCOMOTION}" \
  seed=2000 seeds=32 duration=10.0 naconmax=256 njmax=4096 \
  stochastic_actions="${STOCHASTIC}" record_trajectories=false \
  output="${OUTPUT}"
