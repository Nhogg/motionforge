#!/usr/bin/env bash
# One launcher, three shaping strengths x three paired seeds; no trainer forks.
# Run through scripts/slurm/prod.sbatch; see docs/shaping_ablation.md.
# Resource requests belong in the sbatch command, not this worker.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export UV_PROJECT_ENVIRONMENT=.venv-mjlab
STAGE1=${1:?Supply an absolute mjlab stage-1 checkpoint path}
[[ ${STAGE1} = /* && -f ${STAGE1} ]] || { echo 'Missing absolute stage-1 checkpoint; refusing a zero-action opponent.' >&2; exit 2; }
TASK=${SLURM_ARRAY_TASK_ID:?}
WEIGHTS=(0.0 0.25 1.0)
WEIGHT=${WEIGHTS[$((TASK / 3))]}; SEED=$((TASK % 3)); ENABLED=true
if [[ ${TASK} -lt 3 ]]; then ENABLED=false; fi
ITERATIONS=1000; ENVS=4096; EVAL_ENVS=256; SAVE=250
if [[ ${SMOKE:-0} == 1 ]]; then ITERATIONS=2; ENVS=16; EVAL_ENVS=6; SAVE=1; fi
OUT="logs/mofo/shaping_${SLURM_ARRAY_JOB_ID}/w${WEIGHT}_s${SEED}"
# Sync once on the login node before submission; compute-node PATH need not have uv.
PY=("${PWD}/.venv-mjlab/bin/python")
[[ -x ${PY[0]} ]] || { echo 'Run UV_PROJECT_ENVIRONMENT=.venv-mjlab uv sync --locked first.' >&2; exit 2; }
COMMON=(+experiment=shaping "shaping.enabled=${ENABLED}" "shaping.weight=${WEIGHT}")
"${PY[@]}" scripts/mofo_train.py tag "${COMMON[@]}" "seed=${SEED}" \
  "cmd.init=${STAGE1}" "cmd.logdir=${OUT}" "num_envs=${ENVS}" \
  "cmd.iterations=${ITERATIONS}" "agent.save_interval=${SAVE}"
# Initial and final policies see identical evaluation seeds/resets/opponents.
# One trial per physical world, deterministic actors, post-TAG physics continuation.
CHECKPOINTS=(initial "$((ITERATIONS-1))")
if [[ ${SMOKE:-0} != 1 ]]; then CHECKPOINTS=(initial 250 500 "$((ITERATIONS-1))"); fi
for STEP in "${CHECKPOINTS[@]}"; do
  POLICY="${OUT}/model_${STEP}.pt"
  if [[ ${STEP} == initial ]]; then POLICY="${OUT}/initial.pt"; fi
  for MATCHUP in scripted selfplay; do
    OPPONENT=("opponent=${STAGE1}" cmd.opponent_policy=null)
    RESULT="${OUT}/eval/model_${STEP}.json"
    if [[ ${STEP} == initial ]]; then RESULT="${OUT}/eval/initial.json"; fi
    if [[ ${MATCHUP} == selfplay ]]; then
      OPPONENT=("cmd.opponent_policy=${POLICY}")
      RESULT="${OUT}/eval/selfplay/model_${STEP}.json"
      if [[ ${STEP} == initial ]]; then RESULT="${OUT}/eval/selfplay/initial.json"; fi
    fi
    "${PY[@]}" scripts/mofo_train.py eval "${COMMON[@]}" "${OPPONENT[@]}" \
      seed=30000 "num_envs=${EVAL_ENVS}" "cmd.policy=${POLICY}" \
      "cmd.out=${RESULT}" cmd.post_tag_seconds=1.0
  done
done
# One predeclared seed-0 self-play example per arm; stats above use all trials.
if [[ ${SEED} == 0 && ${SMOKE:-0} != 1 ]]; then
  "${PY[@]}" scripts/mofo_train.py eval "${COMMON[@]}" seed=30000 num_envs=1 \
    "cmd.policy=${POLICY}" "cmd.opponent_policy=${POLICY}" \
    "cmd.out=${OUT}/eval/final_video.json" "cmd.video=${OUT}/eval/final_selfplay.mp4"
fi
