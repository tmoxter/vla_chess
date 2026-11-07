#!/usr/bin/env bash
set -euo pipefail

# A policy's baseline on the scene as it is now built, on a GPU pod: a sweep over the trained
# one-step moves with every frame's simulator state recorded (the primary metric, and rollouts a
# later round can take over or learn from), a sweep over held-out one-step moves, and the standard
# d7->d6 eval. With DART_OUT set it then regenerates the DART funnel dataset on the same scene.
# Every step is skipped when its output already exists, so rerunning the script resumes it. Prints
# BASELINE_EXIT_<code> at the end for polling.
#
# Knobs come from the environment, or from /root/run.env (the ENV_FILE bootstrap.sh copies over).

[[ -f /root/run.env ]] && { set -a; source /root/run.env; set +a; }

POD_REPO="${POD_REPO:-/workspace/vla_chess}"
EVAL_VENV="${EVAL_VENV:-/root/eval-venv}"
OUT_DIR="${OUT_DIR:?set OUT_DIR, e.g. /workspace/baseline/round3}"
POLICY_IN="${POLICY_IN:?set POLICY_IN to a pretrained_model directory}"
WORKERS="${WORKERS:-4}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
SWEEP_ROLLOUTS="${SWEEP_ROLLOUTS:-1000}"
# Held-out one-step pushes, as source:target pairs; empty skips the sweep.
HELDOUT_MOVES="${HELDOUT_MOVES-d6:d5,e5:e4}"
HELDOUT_ROLLOUTS="${HELDOUT_ROLLOUTS:-200}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-1000}"
# Where to write the regenerated DART dataset; empty skips it.
DART_OUT="${DART_OUT:-}"
DART_REPO_ID="${DART_REPO_ID:-vla_chess/dart}"

export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/baseline.log"
exec > >(tee -a "${LOG}") 2>&1
trap 'echo "BASELINE_EXIT_$?"' EXIT
step() { echo; echo ">> [$(date -u +%H:%M:%S)] $*"; }
cd "${POD_REPO}"

if [[ ! -x "${EVAL_VENV}/bin/python" ]]; then
  step "eval/datagen env missing: building it"
  REPO="${POD_REPO}" VENV="${EVAL_VENV}" bash "${POD_REPO}/runpod/eval_env_setup.sh"
fi
PY="${EVAL_VENV}/bin/python"
common=(--local-checkpoint "${POLICY_IN}" --fps 30 --n-action-steps "${N_ACTION_STEPS}"
        --workers "${WORKERS}" --device cuda)

step "1/4 sweep: trained one-step moves, ${SWEEP_ROLLOUTS} rollouts with state recording"
if [[ ! -f "${OUT_DIR}/sweep-trained.json" ]]; then
  "${PY}" -m vla_chess.cli.eval_metrics "${common[@]}" --trained-one-step \
    --rollouts "${SWEEP_ROLLOUTS}" --no-reference --record-states "${OUT_DIR}/rollouts" \
    --out "${OUT_DIR}/sweep-trained.json"
else
  echo "   done already"
fi

step "2/4 sweep: held-out moves ${HELDOUT_MOVES:-(none)}, ${HELDOUT_ROLLOUTS} rollouts"
if [[ -z "${HELDOUT_MOVES}" ]]; then
  echo "   skipped (HELDOUT_MOVES empty)"
elif [[ ! -f "${OUT_DIR}/sweep-heldout.json" ]]; then
  "${PY}" -m vla_chess.cli.eval_metrics "${common[@]}" --moves "${HELDOUT_MOVES}" \
    --rollouts "${HELDOUT_ROLLOUTS}" --no-reference --out "${OUT_DIR}/sweep-heldout.json"
else
  echo "   done already"
fi

step "3/4 eval: d7->d6, ${EVAL_ROLLOUTS} rollouts"
if [[ ! -f "${OUT_DIR}/eval-d7d6.json" ]]; then
  "${PY}" -m vla_chess.cli.eval_metrics "${common[@]}" --source d7 --target d6 \
    --rollouts "${EVAL_ROLLOUTS}" --out "${OUT_DIR}/eval-d7d6.json"
else
  echo "   done already"
fi

step "4/4 DART funnel dataset to ${DART_OUT:-(none)}"
if [[ -z "${DART_OUT}" ]]; then
  echo "   skipped (DART_OUT empty)"
elif [[ ! -f "${DART_OUT}/meta/info.json" ]]; then
  PYTHON="${PY}" OUT="${DART_OUT}" REPO_ID="${DART_REPO_ID}" bash "${POD_REPO}/runpod/dart_datagen.sh"
else
  echo "   done already"
fi
