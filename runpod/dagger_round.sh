#!/usr/bin/env bash
set -euo pipefail

# One DAgger round on a GPU pod: roll the current policy out over the trained one-step moves,
# recording every frame's simulator state; let the scripted expert take those rollouts over and
# keep the recoveries that place the pawn; merge them with the datasets trained on so far; fine-tune
# the policy on the union; evaluate it with the standard d7->d6 sweep and, when SWEEP_ROLLOUTS is
# set, again over the trained one-step moves the round's rollouts drove; when BASELINE_ROLLOUTS is
# set, score the starting policy on d7->d6 with the same eval build. Every step is skipped when
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
# CARRY_LEAD before a mid-air release, then up to CARRY_PER_TIER other held frames, then the rest;
# release: only the last RELEASE_LEAD held frames before the jaws were opened in mid-air;
# placed: only frames after the pawn was placed and before the arm was back at rest.
FRAME_ORDER="${FRAME_ORDER:?set FRAME_ORDER to random, carry, release or placed}"
CARRY_LEAD="${CARRY_LEAD:-45}"
CARRY_PER_TIER="${CARRY_PER_TIER:-3}"
RELEASE_LEAD="${RELEASE_LEAD:-30}"
# Seconds every takeover holds the arm still at rest after flying home.
REST_HOLD="${REST_HOLD:-0}"
# One takeover shard per CPU the container may use, as far as its memory allows: nproc reports
# the host's cores, while the cgroup quotas are what the shards actually get. Containers expose
# the quotas through cgroup v2 (cpu.max, memory.max) or v1 (cpu.cfs_quota_us, memory.limit_in_bytes,
# where -1 and a near-2^63 limit mean unlimited). A shard holds a whole rollout's snapshots and a
# renderer, about 2 GB resident.
SHARD_MEM_GB="${SHARD_MEM_GB:-3}"
cpu_quota() {
  local quota period
  if [[ -r /sys/fs/cgroup/cpu.max ]] && read -r quota period </sys/fs/cgroup/cpu.max \
      && [[ "${quota}" != "max" ]]; then
    :
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]] \
      && read -r quota </sys/fs/cgroup/cpu/cpu.cfs_quota_us && [[ "${quota}" != "-1" ]] \
      && read -r period </sys/fs/cgroup/cpu/cpu.cfs_period_us; then
    :
  else
    nproc
    return
  fi
  echo $(( quota / period > 0 ? quota / period : 1 ))
}
memory_shards() {
  local limit host
  host=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) * 1024 ))
  if [[ -r /sys/fs/cgroup/memory.max ]] && read -r limit </sys/fs/cgroup/memory.max \
      && [[ "${limit}" != "max" ]]; then
    :
  elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]] \
      && read -r limit </sys/fs/cgroup/memory/memory.limit_in_bytes && (( limit < host )); then
    :
  else
    limit="${host}"
  fi
  echo $(( limit / (SHARD_MEM_GB * 1024 ** 3) > 0 ? limit / (SHARD_MEM_GB * 1024 ** 3) : 1 ))
}
cpus="$(cpu_quota)"
fits="$(memory_shards)"
SHARDS="${SHARDS:-$(( cpus < fits ? cpus : fits ))}"
SEED="${SEED:-0}"
STEPS="${STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-1000}"
# Rollouts of the trained policy over the trained one-step moves, to compare with step 1's; 0 skips.
SWEEP_ROLLOUTS="${SWEEP_ROLLOUTS:-0}"
# d7->d6 rollouts of the starting policy, scored by the same eval build as the trained one; 0 skips.
BASELINE_ROLLOUTS="${BASELINE_ROLLOUTS:-0}"
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

step "1/7 policy rollouts with state recording (${ROLLOUTS} over the trained one-step moves)"
if [[ ! -f "${ROUND_DIR}/rollouts.json" ]]; then
  "${PY}" -m vla_chess.cli.eval_metrics --local-checkpoint "${POLICY_IN}" --trained-one-step \
    --fps 30 --n-action-steps "${N_ACTION_STEPS}" --rollouts "${ROLLOUTS}" --workers "${WORKERS}" \
    --device cuda --no-reference --record-states "${ROUND_DIR}/rollouts" \
    --out "${ROUND_DIR}/rollouts.json"
else
  echo "   done already"
fi

step "2/7 expert takeovers over ${SHARDS} shards (${PER_ROLLOUT} per rollout, ${FRAME_ORDER} frame order)"
if [[ ! -f "${ROUND_DIR}/takeovers/meta/info.json" ]]; then
  common=(--rollouts "${ROUND_DIR}/rollouts" --captures "${ROUND_DIR}/takeover-captures"
          --per-rollout "${PER_ROLLOUT}" --max-attempts "${MAX_ATTEMPTS}"
          --frame-order "${FRAME_ORDER}" --carry-lead "${CARRY_LEAD}"
          --carry-per-tier "${CARRY_PER_TIER}" --release-lead "${RELEASE_LEAD}"
          --rest-hold "${REST_HOLD}" --fps 30 --seed "${SEED}")
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

step "3/7 merge ${BASE_DATASETS} + this round's takeovers"
if [[ ! -f "${ROUND_DIR}/train-dataset/meta/info.json" ]]; then
  # shellcheck disable=SC2086
  "${PY}" -m vla_chess.cli.merge_datasets --out "${ROUND_DIR}/train-dataset" \
    --repo-id "${DATASET_REPO_ID}" ${BASE_DATASETS} "${ROUND_DIR}/takeovers"
else
  echo "   done already"
fi

final_step="$(printf '%06d' "${STEPS}")"
CHECKPOINT="${ROUND_DIR}/train/checkpoints/${final_step}/pretrained_model"
step "4/7 fine-tune ${POLICY_IN} for ${STEPS} steps"
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

step "5/7 eval: d7->d6, ${EVAL_ROLLOUTS} rollouts"
if [[ ! -f "${ROUND_DIR}/eval-d7d6.json" ]]; then
  "${PY}" -m vla_chess.cli.eval_metrics --local-checkpoint "${CHECKPOINT}" --source d7 --target d6 \
    --fps 30 --n-action-steps "${N_ACTION_STEPS}" --rollouts "${EVAL_ROLLOUTS}" \
    --workers "${WORKERS}" --device cuda --out "${ROUND_DIR}/eval-d7d6.json"
else
  echo "   done already"
fi

step "6/7 sweep: trained one-step moves, ${SWEEP_ROLLOUTS} rollouts"
if [[ "${SWEEP_ROLLOUTS}" == 0 ]]; then
  echo "   skipped (SWEEP_ROLLOUTS=0)"
elif [[ ! -f "${ROUND_DIR}/sweep-trained.json" ]]; then
  "${PY}" -m vla_chess.cli.eval_metrics --local-checkpoint "${CHECKPOINT}" --trained-one-step \
    --fps 30 --n-action-steps "${N_ACTION_STEPS}" --rollouts "${SWEEP_ROLLOUTS}" \
    --workers "${WORKERS}" --device cuda --no-reference --out "${ROUND_DIR}/sweep-trained.json"
else
  echo "   done already"
fi

step "7/7 baseline: the starting policy on d7->d6, ${BASELINE_ROLLOUTS} rollouts"
if [[ "${BASELINE_ROLLOUTS}" == 0 ]]; then
  echo "   skipped (BASELINE_ROLLOUTS=0)"
elif [[ ! -f "${ROUND_DIR}/baseline-d7d6.json" ]]; then
  "${PY}" -m vla_chess.cli.eval_metrics --local-checkpoint "${POLICY_IN}" --source d7 --target d6 \
    --fps 30 --n-action-steps "${N_ACTION_STEPS}" --rollouts "${BASELINE_ROLLOUTS}" \
    --workers "${WORKERS}" --device cuda --out "${ROUND_DIR}/baseline-d7d6.json"
else
  echo "   done already"
fi
