#!/usr/bin/env bash
#SBATCH --job-name=tag-pursuit-seeds
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
GAE=0.95
case "${TASK_ID}" in
  0|1|2|3|4) SEED="$((TASK_ID + 2))"; VARIANT=fresh ;;
  5) SEED=0; VARIANT=full_credit; GAE=1.0 ;;
  6) SEED=1; VARIANT=full_credit; GAE=1.0 ;;
  7) SEED=0; VARIANT=control ;;
  *) echo "Unsupported task ${TASK_ID}" >&2; exit 2 ;;
esac

LOCOMOTION=logs/p2/training/g1_structured_finetune_40m_seed0/checkpoints/000040632320
HBASE=logs/hierarchical/training/tag_sparse_terminal_640k_seed0/checkpoints
HISTORICAL="[$HBASE/000000000005,$HBASE/000000000010,$HBASE/000000000015,$HBASE/000000000020]"
for CHECKPOINT in "${LOCOMOTION}" "${HBASE}"/{000000000005,000000000010,000000000015,000000000020}; do
  if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
  fi
done

RUN="tag_pursuit_followup_${VARIANT}_seed${SEED}_job${SLURM_ARRAY_JOB_ID}"
OUTPUT="logs/hierarchical/training/${RUN}"
mkdir -p logs/slurm logs/hierarchical/training
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} task=${TASK_ID} run=${RUN} strategy_gae_lambda=${GAE}"

# Fresh strategy + pretrained P2 locomotion. No restored strategy or optimizer.
# Changing GAE lambda changes credit assignment, not the task payoff.
uv run scripts/training/train_hierarchical_tag.py \
  locomotion_checkpoint="${LOCOMOTION}" \
  "historical_opponent_checkpoints=${HISTORICAL}" \
  historical_opponent_probability=0.75 opponent_mode=warm_start \
  freeze_locomotion=true save_initial_checkpoint=true \
  strategy_gae_lambda="${GAE}" strategy_discount=1.0 \
  minimum_initial_separation=1.5 maximum_initial_separation=1.5 \
  strategy_learning_rate=3e-4 locomotion_learning_rate=1e-5 \
  seed="${SEED}" num_environments=4096 minibatch_size=1024 \
  num_iterations=100 rollout_horizon=500 episode_duration=10.0 \
  checkpoint_interval=25 naconmax_per_environment=64 njmax=4096 \
  wandb_mode=online wandb_group=hierarchical_tag_pursuit_followup \
  wandb_name="${RUN}" wandb_videos=true \
  wandb_video_interval_steps=51200000 wandb_video_duration=10.0 \
  output_dir="${OUTPUT}"

# Matched before/after evaluation against development opponents, both roles.
for ITERATION in 000000000000 000000000100; do
  for OPPONENT_ITERATION in 000000000005 000000000020; do
    for STOCHASTIC in false true; do
      uv run scripts/evaluation/evaluate_hierarchical_tag.py \
        checkpoint="${OUTPUT}/checkpoints/${ITERATION}" \
        opponent_checkpoint="${HBASE}/${OPPONENT_ITERATION}" \
        locomotion_checkpoint="${LOCOMOTION}" \
        seed=1000 seeds=16 duration=10.0 naconmax=256 njmax=4096 \
        stochastic_actions="${STOCHASTIC}" record_trajectories=false \
        output="${OUTPUT}/evaluation/iter_${ITERATION}/vs_${OPPONENT_ITERATION}_stochastic_${STOCHASTIC}.json"
    done
  done
done
