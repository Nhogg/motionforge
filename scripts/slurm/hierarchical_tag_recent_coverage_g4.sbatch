#!/usr/bin/env bash
#SBATCH --job-name=tag-recent-coverage
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
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
PAIR=$((TASK_ID / 2))
case "${PAIR}" in
  0) SEED=1; SOURCE=tag_pursuit_diag_mixed_frozen_seed1_job4199 ;;
  1) SEED=2; SOURCE=tag_pursuit_followup_fresh_seed2_job4234 ;;
  2) SEED=4; SOURCE=tag_pursuit_followup_fresh_seed4_job4234 ;;
  3) SEED=5; SOURCE=tag_pursuit_followup_fresh_seed5_job4234 ;;
  *) echo "Unsupported task ${TASK_ID}" >&2; exit 2 ;;
esac
VARIANT=control
HISTORICAL_PROBABILITY=0.75
RECENT_PROBABILITY=null
if (( TASK_ID % 2 == 1 )); then
  VARIANT=recent50
  HISTORICAL_PROBABILITY=0.25
  RECENT_PROBABILITY=0.5
fi
RESTORE="logs/hierarchical/training/${SOURCE}/checkpoints/000000000100"
LOCOMOTION=logs/p2/training/g1_rebuilt_full_seed0/checkpoints/000200048640
HBASE=logs/hierarchical/training/tag_sparse_terminal_640k_seed0/checkpoints
HISTORICAL="[$HBASE/000000000005,$HBASE/000000000010,$HBASE/000000000015,$HBASE/000000000020]"
for CHECKPOINT in "${RESTORE}" "${LOCOMOTION}" "${HBASE}"/{000000000005,000000000010,000000000015,000000000020}; do
  [[ -d "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
done
RUN="tag_recent_coverage_${VARIANT}_seed${SEED}_job${SLURM_ARRAY_JOB_ID}"
OUTPUT="logs/hierarchical/training/${RUN}"
PYTHON=/home/nhogg/motionforge/.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} source=${SOURCE} variant=${VARIANT}"

# Same starting parameters AND optimizer state within each pair; only mixture differs.
"${PYTHON}" scripts/training/train_hierarchical_tag.py \
  locomotion_checkpoint="${LOCOMOTION}" restore_checkpoint="${RESTORE}" \
  "historical_opponent_checkpoints=${HISTORICAL}" \
  historical_opponent_probability="${HISTORICAL_PROBABILITY}" \
  recent_opponent_probability="${RECENT_PROBABILITY}" opponent_mode=warm_start \
  freeze_locomotion=true save_initial_checkpoint=true \
  strategy_gae_lambda=0.95 strategy_discount=1.0 \
  minimum_initial_separation=1.5 maximum_initial_separation=1.5 \
  strategy_learning_rate=3e-4 locomotion_learning_rate=1e-5 \
  seed="${SEED}" num_environments=4096 minibatch_size=1024 \
  num_iterations=100 rollout_horizon=500 episode_duration=10.0 \
  checkpoint_interval=25 naconmax_per_environment=64 njmax=4096 \
  wandb_mode=online wandb_project=MotionForge wandb_entity=luc-ssl \
  wandb_group=hierarchical_tag_recent_coverage wandb_name="${RUN}" \
  wandb_videos=true wandb_video_interval_steps=51200000 wandb_video_duration=10.0 \
  output_dir="${OUTPUT}"

# Both action modes and both roles, identical held-out reset seeds and separation.
for ITERATION in 000000000100 000000000200; do
  for STOCHASTIC in false true; do
    "${PYTHON}" scripts/evaluation/evaluate_hierarchical_tag.py \
      checkpoint="${OUTPUT}/checkpoints/${ITERATION}" use_self_play=true \
      initial_separation=1.5 locomotion_checkpoint="${LOCOMOTION}" \
      seed=2000 seeds=32 duration=10.0 episode_duration=10.0 naconmax=256 njmax=4096 \
      stochastic_actions="${STOCHASTIC}" record_trajectories=false \
      output="${OUTPUT}/evaluation/iter_${ITERATION}/selfplay_stochastic_${STOCHASTIC}.json"
    for OPPONENT_ITERATION in 000000000005 000000000020; do
      "${PYTHON}" scripts/evaluation/evaluate_hierarchical_tag.py \
        checkpoint="${OUTPUT}/checkpoints/${ITERATION}" \
        opponent_checkpoint="${HBASE}/${OPPONENT_ITERATION}" initial_separation=1.5 \
        locomotion_checkpoint="${LOCOMOTION}" \
        seed=2000 seeds=16 duration=10.0 episode_duration=10.0 naconmax=256 njmax=4096 \
        stochastic_actions="${STOCHASTIC}" record_trajectories=false \
        output="${OUTPUT}/evaluation/iter_${ITERATION}/historical${OPPONENT_ITERATION}_stochastic_${STOCHASTIC}.json"
    done
  done
done
