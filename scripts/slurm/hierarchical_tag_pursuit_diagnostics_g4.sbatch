#!/usr/bin/env bash
#SBATCH --job-name=tag-pursuit-diag
#SBATCH --partition=gpu
#SBATCH --nodelist=g4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
cd /home/nhogg/motionforge

# Four conditions x seeds 0 and 1; each task uses one Slurm-assigned GPU.
# All runs start fresh from the same P2 locomotion checkpoint, at 1.5 m.
# 100 iterations x 4096 environments x 500 steps = 204,800,000 steps.
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this file using sbatch}"
SEED="$((TASK_ID % 2))"
VARIANT_INDEX="$((TASK_ID / 2))"
FREEZE=false
OPPONENT_MODE=warm_start
LEAGUE_ARGS=()
LOCOMOTION=logs/p2/training/g1_rebuilt_full_seed0/checkpoints/000200048640
HBASE=logs/hierarchical/training/tag_sparse_terminal_640k_seed0/checkpoints
HISTORICAL="[$HBASE/000000000005,$HBASE/000000000010,$HBASE/000000000015,$HBASE/000000000020]"

case "${VARIANT_INDEX}" in
  0) VARIANT=stationary_joint; OPPONENT_MODE=stationary ;;
  1) VARIANT=stationary_frozen; OPPONENT_MODE=stationary; FREEZE=true ;;
  2) VARIANT=mixed_joint ;;
  3) VARIANT=mixed_frozen; FREEZE=true ;;
  *) echo "Unsupported task ${TASK_ID}" >&2; exit 2 ;;
esac
if [[ "${OPPONENT_MODE}" == warm_start ]]; then
  LEAGUE_ARGS=("historical_opponent_checkpoints=${HISTORICAL}" historical_opponent_probability=0.75)
fi

# Check both training and post-training evaluation inputs before allocating work.
for CHECKPOINT in "${LOCOMOTION}" "${HBASE}"/{000000000005,000000000010,000000000015,000000000020}; do
  if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
  fi
done
RUN="tag_pursuit_diag_${VARIANT}_seed${SEED}_job${SLURM_ARRAY_JOB_ID}"
OUTPUT="logs/hierarchical/training/${RUN}"
mkdir -p logs/slurm logs/hierarchical/training

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} task=${TASK_ID} run=${RUN} freeze_locomotion=${FREEZE}"

uv run scripts/training/train_hierarchical_tag.py \
  locomotion_checkpoint="${LOCOMOTION}" \
  "${LEAGUE_ARGS[@]}" \
  opponent_mode="${OPPONENT_MODE}" \
  freeze_locomotion="${FREEZE}" \
  minimum_initial_separation=1.5 maximum_initial_separation=1.5 \
  strategy_learning_rate=3e-4 locomotion_learning_rate=1e-5 \
  seed="${SEED}" num_environments=4096 minibatch_size=1024 \
  num_iterations=100 rollout_horizon=500 episode_duration=10.0 \
  checkpoint_interval=25 naconmax_per_environment=64 njmax=4096 \
  wandb_mode=online wandb_group=hierarchical_tag_pursuit_diagnostics \
  wandb_name="${RUN}" wandb_videos=true \
  wandb_video_interval_steps=51200000 wandb_video_duration=10.0 \
  output_dir="${OUTPUT}"

# Fixed opponents and identical evaluation seeds make comparisons meaningful.
# Evaluate both learner roles under deterministic and stochastic actions.
for OPPONENT_ITERATION in 000000000005 000000000020; do
  for STOCHASTIC in false true; do
    uv run scripts/evaluation/evaluate_hierarchical_tag.py \
      checkpoint="${OUTPUT}/checkpoints/000000000100" \
      opponent_checkpoint="${HBASE}/${OPPONENT_ITERATION}" \
      locomotion_checkpoint="${LOCOMOTION}" \
      seed=1000 seeds=16 duration=10.0 naconmax=256 njmax=4096 \
      stochastic_actions="${STOCHASTIC}" record_trajectories=false \
      output="${OUTPUT}/evaluation/vs_${OPPONENT_ITERATION}_stochastic_${STOCHASTIC}.json"
  done
done
