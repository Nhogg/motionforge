#!/usr/bin/env bash
#SBATCH --job-name=tag-potential-diagnostic
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
TASK="${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
SEED=$((TASK / 2))
PROGRESS=false
VARIANT=no_progress
if (( TASK % 2 == 1 )); then
  PROGRESS=true
  VARIANT=potential025
fi
PYTHON=/home/nhogg/motionforge/.venv/bin/python
LOCOMOTION=logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320
HBASE=logs/hierarchical/training/tag_sparse_terminal_640k_seed0/checkpoints
HISTORICAL="[$HBASE/000000000005,$HBASE/000000000010,$HBASE/000000000015,$HBASE/000000000020]"
for CHECKPOINT in "${LOCOMOTION}" "${HBASE}"/{000000000005,000000000010,000000000015,000000000020}; do
  [[ -d "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
done
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
RUN="tag_potential_diagnostic_${VARIANT}_seed${SEED}_job${SLURM_ARRAY_JOB_ID}"
OUTPUT="logs/hierarchical/training/${RUN}"
# Fresh strategy, unchanged role heads/encoder and pretrained frozen locomotion.
# Both arms use the same truncation/failure/smoothness/boundary semantics; only
# the symmetric progress term differs. Existing historical mixture is retained.
"${PYTHON}" scripts/training/train_hierarchical_tag.py \
  locomotion_checkpoint="${LOCOMOTION}" \
  "historical_opponent_checkpoints=${HISTORICAL}" \
  historical_opponent_probability=0.75 recent_opponent_probability=null \
  freeze_locomotion=true save_initial_checkpoint=true \
  "tag_reward.progress.enabled=${PROGRESS}" tag_reward.progress.alpha=0.25 \
  tag_reward.progress.distance_normalization_m=null tag_reward.command_smoothness_weight=0.005 \
  tag_reward.own_fall_penalty=1.0 tag_reward.own_oob_penalty=1.0 \
  minimum_initial_separation=1.5 maximum_initial_separation=1.5 \
  seed="${SEED}" num_environments=4096 minibatch_size=1024 \
  num_iterations=100 rollout_horizon=500 episode_duration=10.0 \
  strategy_learning_rate=3e-4 locomotion_learning_rate=1e-5 \
  strategy_update_interval=1 locomotion_update_interval=5 strategy_gae_lambda=0.95 \
  checkpoint_interval=25 naconmax_per_environment=64 njmax=4096 \
  wandb_mode=online wandb_project=MotionForge wandb_entity=luc-ssl \
  wandb_group=hierarchical_tag_potential_diagnostic wandb_name="${RUN}" \
  wandb_videos=true wandb_video_interval_steps=51200000 wandb_video_duration=10.0 \
  output_dir="${OUTPUT}"
for ITERATION in 000000000000 000000000100; do
  for STOCHASTIC in false true; do
    "${PYTHON}" scripts/evaluation/evaluate_hierarchical_tag.py \
      checkpoint="${OUTPUT}/checkpoints/${ITERATION}" use_self_play=true \
      initial_separation=1.5 episode_duration=10.0 duration=10.0 \
      seed=3000 seeds=16 njmax=4096 stochastic_actions="${STOCHASTIC}" \
      record_trajectories=false \
      output="${OUTPUT}/evaluation/iter_${ITERATION}/selfplay_stochastic_${STOCHASTIC}.json"
    for OPPONENT in 000000000005 000000000020; do
      "${PYTHON}" scripts/evaluation/evaluate_hierarchical_tag.py \
        checkpoint="${OUTPUT}/checkpoints/${ITERATION}" \
        opponent_checkpoint="${HBASE}/${OPPONENT}" \
        initial_separation=1.5 episode_duration=10.0 duration=10.0 \
        seed=3000 seeds=16 njmax=4096 stochastic_actions="${STOCHASTIC}" \
        record_trajectories=false \
        output="${OUTPUT}/evaluation/iter_${ITERATION}/historical${OPPONENT}_stochastic_${STOCHASTIC}.json"
    done
  done
done
