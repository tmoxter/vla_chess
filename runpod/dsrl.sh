#!/usr/bin/env bash
set -euo pipefail

# One DSRL run on a GPU pod: train a steering agent over a frozen SmolVLA checkpoint on one pawn
# move (python -m vla_chess.cli.train_dsrl), then evaluate the last agent and the frozen policy on
# FINAL_EVAL_EPISODES evaluation starts. Each step is skipped when its output already exists, so
# rerunning the script after a finished training only redoes the final evaluation. Prints
# DSRL_EXIT_<code> at the end for polling.
#
# Knobs come from the environment, or from /root/run.env (the ENV_FILE bootstrap.sh copies over).

[[ -f /root/run.env ]] && { set -a; source /root/run.env; set +a; }

POD_REPO="${POD_REPO:-/workspace/vla_chess}"
EVAL_VENV="${EVAL_VENV:-/root/eval-venv}"
OUT_DIR="${OUT_DIR:?set OUT_DIR, e.g. /workspace/dsrl/v1}"
POLICY_IN="${POLICY_IN:?set POLICY_IN to a pretrained_model directory}"
SOURCE="${SOURCE:-d7}"
TARGET="${TARGET:-d6}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
PIECE_JITTER_MM="${PIECE_JITTER_MM:-3}"
STEERED_STEPS="${STEERED_STEPS:-10}"
MAGNITUDE="${MAGNITUDE:-1.5}"
FILL="${FILL:-zero}"
WARMUP_EPISODES="${WARMUP_EPISODES:-20}"
TRAIN_EPISODES="${TRAIN_EPISODES:-1000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LEARN_AFTER="${LEARN_AFTER:-256}"
UPDATES_PER_CHUNK="${UPDATES_PER_CHUNK:-1}"
HIDDEN="${HIDDEN:-512,512,512}"
NUM_CRITICS="${NUM_CRITICS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DISCOUNT="${DISCOUNT:-0.99}"
LR="${LR:-3e-4}"
TEMPERATURE_INIT="${TEMPERATURE_INIT:-0.1}"
SEED="${SEED:-0}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-200}"

export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/dsrl.log"
exec > >(tee -a "${LOG}") 2>&1
trap 'echo "DSRL_EXIT_$?"' EXIT
step() { echo; echo ">> [$(date -u +%H:%M:%S)] $*"; }
cd "${POD_REPO}"

if [[ ! -x "${EVAL_VENV}/bin/python" ]]; then
  step "eval/datagen env missing: building it"
  REPO="${POD_REPO}" VENV="${EVAL_VENV}" bash "${POD_REPO}/runpod/eval_env_setup.sh"
fi
PY="${EVAL_VENV}/bin/python"
common=(--checkpoint "${POLICY_IN}" --source "${SOURCE}" --target "${TARGET}" --device cuda
        --n-action-steps "${N_ACTION_STEPS}" --piece-jitter-mm "${PIECE_JITTER_MM}"
        --steered-steps "${STEERED_STEPS}" --magnitude "${MAGNITUDE}" --fill "${FILL}")

step "1/2 train: ${WARMUP_EPISODES} warm-up + ${TRAIN_EPISODES} episodes on ${SOURCE}->${TARGET}"
if [[ ! -f "${OUT_DIR}/train/agent-$(printf '%05d' "${TRAIN_EPISODES}").pt" ]]; then
  rm -rf "${OUT_DIR}/train"
  "${PY}" -m vla_chess.cli.train_dsrl "${common[@]}" --out "${OUT_DIR}/train" \
    --warmup-episodes "${WARMUP_EPISODES}" --train-episodes "${TRAIN_EPISODES}" \
    --eval-every "${EVAL_EVERY}" --eval-episodes "${EVAL_EPISODES}" \
    --learn-after "${LEARN_AFTER}" --updates-per-chunk "${UPDATES_PER_CHUNK}" \
    --hidden "${HIDDEN}" --num-critics "${NUM_CRITICS}" --batch-size "${BATCH_SIZE}" \
    --discount "${DISCOUNT}" --lr "${LR}" --temperature-init "${TEMPERATURE_INIT}" --seed "${SEED}"
else
  echo "   done already"
fi

step "2/2 final eval: last agent and the frozen policy, ${FINAL_EVAL_EPISODES} starts each"
if [[ ! -f "${OUT_DIR}/final-eval/evals.jsonl" ]] \
    || [[ "$(wc -l <"${OUT_DIR}/final-eval/evals.jsonl")" -lt 2 ]]; then
  rm -rf "${OUT_DIR}/final-eval"
  "${PY}" -m vla_chess.cli.train_dsrl "${common[@]}" --out "${OUT_DIR}/final-eval" --eval-only \
    --agent "${OUT_DIR}/train/agent-last.pt" --eval-episodes "${FINAL_EVAL_EPISODES}"
else
  echo "   done already"
fi
