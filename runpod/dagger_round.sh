#!/usr/bin/env bash
set -euo pipefail

# One DAgger round on a GPU pod: roll the current policy out over the trained one-step moves,
# recording every frame's simulator state; let the scripted expert take those rollouts over and
# keep the recoveries that place the pawn; merge them with the datasets trained on so far; fine-tune
# the policy on the union; evaluate it with the standard d7->d6 sweep. Every step is skipped when
# its output already exists, so rerunning the script resumes the round. Prints DAGGER_EXIT_<code>
# at the end for polling.
#
# Knobs come from the environment, or from /root/run.env (the ENV_FILE bootstrap.sh copies over).

[[ -f /root/run.env ]] && { set -a; source /root/run.env; set +a; }

POD_REPO="${POD_REPO:-/workspace/vla_chess}"
EVAL_VENV="${EVAL_VENV:-/root/eval-venv}"
TRAIN_VENV="${VENV:-/root/vla-venv}"
ROUND_DIR="${ROUND_DIR:?set ROUND_DIR, e.g. /workspace/dagger/round1}"
# The policy that drives the rollouts and that training starts from.
POLICY_IN="${POLICY_IN:?set POLICY_IN to a pretrained_model directory}"
# Datasets the takeovers are merged with, in order; space-separated roots.
BASE_DATASETS="${BASE_DATASETS:?set BASE_DATASETS to the dataset roots trained on so far}"
ROLLOUTS="${ROLLOUTS:-200}"
WORKERS="${WORKERS:-5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
PER_ROLLOUT="${PER_ROLLOUT:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
# random: take over at any frame; carry: up to CARRY_PER_TIER held frames from the last
# CARRY_LEAD before a mid-air release, then up to CARRY_PER_TIER other held frames, then the rest.
FRAME_ORDER="${FRAME_ORDER:?set FRAME_ORDER to random or carry}"
CARRY_LEAD="${CARRY_LEAD:-45}"
CARRY_PER_TIER="${CARRY_PER_TIER:-3}"
# One takeover shard per CPU the container may use: nproc reports the host's cores, while the
# cgroup quota is what the shards actually get.
cpu_quota() {
  local quota period
  if [[ -r /sys/fs/cgroup/cpu.max ]] && read -r quota period </sys/fs/cgroup/cpu.max \
      && [[ "${quota}" != "max" ]]; then
    echo $(( quota / period > 0 ? quota / period : 1 ))
  else
    nproc
  fi
}
SHARDS="${SHARDS:-$(cpu_quota)}"
SEED="${SEED:-0}"
STEPS="${STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-1000}"
DATASET_REPO_ID="${DATASET_REPO_ID:-vla_chess/dagger}"
POLICY_REPO_ID="${POLICY_REPO_ID:-vla_chess/smolvla-dagger}"
JOB_NAME="${JOB_NAME:-$(basename "${ROUND_DIR}")}"

export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "${ROUND_DIR}"
LOG="${ROUND_DIR}/round.log"
exec > >(tee -a "${LOG}") 2>&1
trap 'echo "DAGGER_EXIT_$?"' EXIT
step() { echo; echo ">> [$(date -u +%H:%M:%S)] $*"; }
cd "${POD_REPO}"

if [[ ! -x "${EVAL_VENV}/bin/python" ]]; then
  step "eval/datagen env missing: building it"
  REPO="${POD_REPO}" VENV="${EVAL_VENV}" bash "${POD_REPO}/runpod/eval_env_setup.sh"
fi
PY="${EVAL_VENV}/bin/python"

step "1/5 policy rollouts with state recording (${ROLLOUTS} over the trained one-step moves)"
if [[ ! -f "${ROUND_DIR}/rollouts.json" ]]; then
  "${PY}" -m vla_chess.cli.eval_metrics --local-checkpoint "${POLICY_IN}" --trained-one-step \
    --fps 30 --n-action-steps "${N_ACTION_STEPS}" --rollouts "${ROLLOUTS}" --workers "${WORKERS}" \
    --device cuda --no-reference --record-states "${ROUND_DIR}/rollouts" \
    --out "${ROUND_DIR}/rollouts.json"
else
  echo "   done already"
fi

step "2/5 expert takeovers over ${SHARDS} shards (${PER_ROLLOUT} per rollout, ${FRAME_ORDER} frame order)"
if [[ ! -f "${ROUND_DIR}/takeovers/meta/info.json" ]]; then
  common=(--rollouts "${ROUND_DIR}/rollouts" --captures "${ROUND_DIR}/takeover-captures"
          --per-rollout "${PER_ROLLOUT}" --max-attempts "${MAX_ATTEMPTS}"
          --frame-order "${FRAME_ORDER}" --carry-lead "${CARRY_LEAD}"
          --carry-per-tier "${CARRY_PER_TIER}" --fps 30 --seed "${SEED}")
  mkdir -p "${ROUND_DIR}/takeover-logs"
  pids=()
  for ((i = 0; i < SHARDS; i++)); do
    "${PY}" -m vla_chess.cli.dagger_takeovers "${common[@]}" --shard "${i}/${SHARDS}" \
      --no-serialize >"${ROUND_DIR}/takeover-logs/shard${i}.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" == 0 ]] || { echo "a takeover shard failed; see ${ROUND_DIR}/takeover-logs"; exit 1; }
  "${PY}" -m vla_chess.cli.dagger_takeovers "${common[@]}" \
    --out "${ROUND_DIR}/takeovers" --repo-id "${DATASET_REPO_ID}-takeovers" 2>&1 | grep -v Svt
else
  echo "   done already"
fi

step "3/5 merge ${BASE_DATASETS} + this round's takeovers"
if [[ ! -f "${ROUND_DIR}/train-dataset/meta/info.json" ]]; then
  # shellcheck disable=SC2086
  "${PY}" -m vla_chess.cli.merge_datasets --out "${ROUND_DIR}/train-dataset" \
    --repo-id "${DATASET_REPO_ID}" ${BASE_DATASETS} "${ROUND_DIR}/takeovers"
else
  echo "   done already"
fi

final_step="$(printf '%06d' "${STEPS}")"
CHECKPOINT="${ROUND_DIR}/train/checkpoints/${final_step}/pretrained_model"
step "4/5 fine-tune ${POLICY_IN} for ${STEPS} steps"
if [[ ! -f "${CHECKPOINT}/model.safetensors" ]]; then
  rm -rf "${ROUND_DIR}/train"
  VENV="${TRAIN_VENV}" POLICY_PATH="${POLICY_IN}" DATASET_REPO_ID="${DATASET_REPO_ID}" \
    DATASET_ROOT="${ROUND_DIR}/train-dataset" POLICY_REPO_ID="${POLICY_REPO_ID}" HUB_PUSH=false \
    STEPS="${STEPS}" SAVE_FREQ="${SAVE_FREQ}" BATCH_SIZE="${BATCH_SIZE}" JOB_NAME="${JOB_NAME}" \
    OUTPUT_DIR="${ROUND_DIR}/train" METRICS_DIR="${ROUND_DIR}/train_logs" WANDB_API_KEY='' \
    bash "${POD_REPO}/runpod/train.sh"
else
  echo "   done already"
fi

step "5/5 eval: d7->d6, ${EVAL_ROLLOUTS} rollouts"
if [[ ! -f "${ROUND_DIR}/eval-d7d6.json" ]]; then
  "${PY}" -m vla_chess.cli.eval_metrics --local-checkpoint "${CHECKPOINT}" --source d7 --target d6 \
    --fps 30 --n-action-steps "${N_ACTION_STEPS}" --rollouts "${EVAL_ROLLOUTS}" \
    --workers "${WORKERS}" --device cuda --out "${ROUND_DIR}/eval-d7d6.json"
else
  echo "   done already"
fi
