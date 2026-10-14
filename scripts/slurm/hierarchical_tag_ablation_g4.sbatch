#!/usr/bin/env bash
#SBATCH --job-name=tag-ablate
#SBATCH --partition=gpu
#SBATCH --nodelist=g3
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%8
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

cd /home/nhogg/motionforge
mkdir -p logs/slurm logs/hierarchical/training

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SEED=0
VARIANT=""
STRATEGY_LR="3e-4"
HISTORICAL_PROBABILITY="0.75"
MINIMUM_SEPARATION="1.5"
MAXIMUM_SEPARATION="3.0"

case "${TASK_ID}" in
  0|1|2)
    VARIANT="strategy_lr_1e3"
    SEED="${TASK_ID}"
    STRATEGY_LR="1e-3"
    ;;
  3|4|5)
    VARIANT="strategy_lr_3e3"
    SEED="$((TASK_ID - 3))"
    STRATEGY_LR="3e-3"
    ;;
  6|7|8)
    VARIANT="historical_025"
    SEED="$((TASK_ID - 6))"
    HISTORICAL_PROBABILITY="0.25"
    ;;
  9|10|11)
    VARIANT="historical_050"
    SEED="$((TASK_ID - 9))"
    HISTORICAL_PROBABILITY="0.50"
    ;;
  12|13|14|15|16)
    VARIANT="historical_050_strategy_lr_1e3"
    SEED="$((TASK_ID - 12))"
    STRATEGY_LR="1e-3"
    HISTORICAL_PROBABILITY="0.50"
    ;;
  17|18|19)
    VARIANT="fixed_separation_1p5"
    SEED="$((TASK_ID - 17))"
    MAXIMUM_SEPARATION="1.5"
    ;;
  *)
    echo "Unsupported array task ${TASK_ID}" >&2
    exit 2
    ;;
esac

HBASE=logs/hierarchical/training/tag_sparse_terminal_640k_seed0/checkpoints
HISTORICAL="[$HBASE/000000000005,$HBASE/000000000010,$HBASE/000000000015,$HBASE/000000000020]"
RUN="tag_ablation_${VARIANT}_seed${SEED}_job${SLURM_ARRAY_JOB_ID}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} task=${TASK_ID} run=${RUN}"

uv run scripts/training/train_hierarchical_tag.py \
  locomotion_checkpoint=logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320 \
  "historical_opponent_checkpoints=${HISTORICAL}" \
  historical_opponent_probability="${HISTORICAL_PROBABILITY}" \
  minimum_initial_separation="${MINIMUM_SEPARATION}" \
  maximum_initial_separation="${MAXIMUM_SEPARATION}" \
  strategy_learning_rate="${STRATEGY_LR}" \
  seed="${SEED}" \
  num_environments=4096 \
  minibatch_size=1024 \
  num_iterations=500 \
  rollout_horizon=500 \
  episode_duration=10.0 \
  checkpoint_interval=50 \
  naconmax_per_environment=64 \
  njmax=4096 \
  wandb_mode=online \
  wandb_group=hierarchical_tag_ablation_screen \
  wandb_name="${RUN}" \
  wandb_videos=true \
  wandb_video_interval_steps=102400000 \
  wandb_video_duration=10.0 \
  elo_baseline_pursuer_win_probability=0.65 \
  output_dir="logs/hierarchical/training/${RUN}"
