#!/usr/bin/env bash
#SBATCH --job-name=tag-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=g3
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-127%8
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

cd /home/nhogg/motionforge
mkdir -p logs/slurm logs/hierarchical/evaluation/ablation_baseline_matrix

TASK_ID="${SLURM_ARRAY_TASK_ID}"
LEARNER_SEED="$((TASK_ID / 8))"
REMAINDER="$((TASK_ID % 8))"
CHECKPOINT_INDEX="$((REMAINDER / 4))"
OPPONENT_INDEX="$((REMAINDER % 4))"

if [[ "${CHECKPOINT_INDEX}" -eq 0 ]]; then
  LEARNER_ITERATION="000000000050"
else
  LEARNER_ITERATION="000000000100"
fi

case "${OPPONENT_INDEX}" in
  0) OPPONENT_ITERATION="000000000005" ;;
  1) OPPONENT_ITERATION="000000000010" ;;
  2) OPPONENT_ITERATION="000000000015" ;;
  3) OPPONENT_ITERATION="000000000020" ;;
esac

LEARNER="logs/hierarchical/training/tag_clean_start_mixed_4096_seed${LEARNER_SEED}_v4/checkpoints/${LEARNER_ITERATION}"
OPPONENT="logs/hierarchical/training/tag_sparse_terminal_640k_seed0/checkpoints/${OPPONENT_ITERATION}"
OUTPUT="logs/hierarchical/evaluation/ablation_baseline_matrix/seed${LEARNER_SEED}_iter${LEARNER_ITERATION}_vs_${OPPONENT_ITERATION}.json"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} learner=${LEARNER} opponent=${OPPONENT}"

uv run scripts/evaluation/evaluate_hierarchical_tag.py \
  checkpoint="${LEARNER}" \
  opponent_checkpoint="${OPPONENT}" \
  locomotion_checkpoint=logs/p2/training/g1_rebuilt_full_seed0/checkpoints/000200048640 \
  seed=0 \
  seeds=16 \
  duration=10.0 \
  naconmax=256 \
  njmax=512 \
  stochastic_actions=false \
  record_trajectories=false \
  output="${OUTPUT}"
