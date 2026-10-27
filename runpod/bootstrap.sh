#!/usr/bin/env bash
set -euo pipefail

# Local helper: copy the pod-side scripts to a running pod over SSH and run them. Reads secrets
# from runpod/.env. Usage:
#   ./bootstrap.sh <ssh_host> <ssh_port>
# where host/port come from `launch.py --create` output.
#
# Optional env:
#   ENV_FILE=runpod/smoke.env   also source this run's knobs (dataset/policy repos, batch, steps,
#                               DATAGEN_ARGS, OUTPUT_DIR, ...) and forward them to the pod.
#   DATAGEN=1                   rsync the repo to the pod and run datagen.sh there (generate + push
#                               the dataset) BEFORE launching training. Needs DATAGEN_ARGS (set by
#                               ENV_FILE). Omit to train against an already-pushed dataset.
#   DATASET_LOCAL=<dir>         upload a dataset generated on this machine to DATASET_ROOT on the
#                               pod and train from it there, with no Hub round-trip.
#   SYNC_REPO=1                 rsync the repo to POD_REPO even without DATAGEN, so the eval
#                               harness (eval_env_setup.sh) can run on the pod afterwards.
#   POLICY_LOCAL=<dir>          upload a pretrained_model directory from this machine to POLICY_IN
#                               on the pod.
#   LAUNCH=dagger               run runpod/dagger_round.sh from the synced repo in the tmux session
#                               instead of train.sh; needs SYNC_REPO=1. The ENV_FILE is copied to
#                               /root/run.env for it to read its knobs from.
#
# End-to-end plumbing smoke test (datagen -> push -> train -> checkpoint push), one command:
#   DATAGEN=1 ENV_FILE=runpod/smoke.env ./runpod/bootstrap.sh <ssh_host> <ssh_port>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
if [[ -z "${SSH_KEY:-}" ]]; then
  SSH_KEY="${HOME}/.ssh/id_ed25519_runpod"
  [[ -f "${SSH_KEY}" ]] || SSH_KEY="${HOME}/.ssh/runpod_ed25519"
fi
POD_REPO="${POD_REPO:-/root/vla_chess}"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <ssh_host> <ssh_port>" >&2
  exit 1
fi
SSH_HOST="$1"
SSH_PORT="$2"

for secrets in "${HERE}/.env" "${REPO_ROOT}/.env"; do
  if [[ -f "${secrets}" ]]; then
    set -a; source "${secrets}"; set +a
    break
  fi
done
# Run-specific knobs (dataset/policy repos, batch size, DATAGEN_ARGS, ...). Sourced after .env so a
# run file can reference secrets, and exported so they forward to the pod-side scripts below.
if [[ -n "${ENV_FILE:-}" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "FATAL: ENV_FILE ${ENV_FILE} not found." >&2
    exit 1
  fi
  set -a; source "${ENV_FILE}"; set +a
fi
if [[ -z "${HF_TOKEN:-}" && ( -z "${DATASET_ROOT:-}" || "${HUB_PUSH:-true}" == "true" ) ]]; then
  echo "FATAL: HF_TOKEN not set (put it in runpod/.env)." >&2
  exit 1
fi
if [[ -n "${DATASET_LOCAL:-}" ]]; then
  if [[ -z "${DATASET_ROOT:-}" ]]; then
    echo "FATAL: DATASET_LOCAL needs DATASET_ROOT, the path it is uploaded to on the pod." >&2
    exit 1
  fi
  if [[ ! -f "${DATASET_LOCAL}/meta/info.json" ]]; then
    echo "FATAL: DATASET_LOCAL=${DATASET_LOCAL} holds no LeRobot dataset." >&2
    exit 1
  fi
fi

if [[ -n "${POLICY_LOCAL:-}" ]]; then
  if [[ -z "${POLICY_IN:-}" ]]; then
    echo "FATAL: POLICY_LOCAL needs POLICY_IN, the path it is uploaded to on the pod." >&2
    exit 1
  fi
  if [[ ! -f "${POLICY_LOCAL}/model.safetensors" ]]; then
    echo "FATAL: POLICY_LOCAL=${POLICY_LOCAL} holds no pretrained_model." >&2
    exit 1
  fi
fi
if [[ "${LAUNCH:-train}" == "dagger" && "${SYNC_REPO:-0}" != "1" ]]; then
  echo "FATAL: LAUNCH=dagger runs the round from the synced repo; set SYNC_REPO=1." >&2
  exit 1
fi

ssh_opts=(-i "${SSH_KEY}" -p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new)

echo ">> copying pod scripts"
scp "${ssh_opts[@]/-p/-P}" \
  "${HERE}/pod_setup.sh" "${HERE}/train.sh" "${HERE}/datagen.sh" \
  "root@${SSH_HOST}:/root/"
if [[ -n "${ENV_FILE:-}" ]]; then
  scp "${ssh_opts[@]/-p/-P}" "${ENV_FILE}" "root@${SSH_HOST}:/root/run.env"
fi

if [[ "${DATAGEN:-0}" == "1" || "${SYNC_REPO:-0}" == "1" || -n "${DATASET_LOCAL:-}" \
      || -n "${POLICY_LOCAL:-}" ]]; then
  echo ">> ensuring rsync is present on the pod"
  ssh "${ssh_opts[@]}" "root@${SSH_HOST}" \
    "command -v rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)"
fi

if [[ "${DATAGEN:-0}" == "1" || "${SYNC_REPO:-0}" == "1" ]]; then
  echo ">> rsyncing source tree to ${POD_REPO} on the pod"
  rsync -az --delete \
    --exclude '.git/' --exclude '.pixi/' --exclude '*-captures/' \
    --exclude 'data/' --exclude 'outputs/' --exclude 'mlflow/' --exclude 'media/' \
    --exclude '.env' --exclude '__pycache__/' --exclude '.*_cache/' --exclude 'runpod/artifacts/' \
    -e "ssh ${ssh_opts[*]}" \
    "${REPO_ROOT}/" "root@${SSH_HOST}:${POD_REPO}/"
fi

if [[ -n "${DATASET_LOCAL:-}" ]]; then
  echo ">> uploading ${DATASET_LOCAL} to ${DATASET_ROOT} on the pod"
  ssh "${ssh_opts[@]}" "root@${SSH_HOST}" "mkdir -p '${DATASET_ROOT}'"
  rsync -az --partial -e "ssh ${ssh_opts[*]}" "${DATASET_LOCAL%/}/" "root@${SSH_HOST}:${DATASET_ROOT}/"
fi

if [[ -n "${POLICY_LOCAL:-}" ]]; then
  echo ">> uploading ${POLICY_LOCAL} to ${POLICY_IN} on the pod"
  ssh "${ssh_opts[@]}" "root@${SSH_HOST}" "mkdir -p '${POLICY_IN}'"
  rsync -az --partial -e "ssh ${ssh_opts[*]}" "${POLICY_LOCAL%/}/" "root@${SSH_HOST}:${POLICY_IN}/"
fi

if [[ "${DATAGEN:-0}" == "1" ]]; then
  if [[ -z "${DATAGEN_ARGS:-}" ]]; then
    echo "FATAL: DATAGEN=1 but DATAGEN_ARGS is empty (set ENV_FILE to smoke.env/exp2.env)." >&2
    exit 1
  fi
  echo ">> running datagen.sh on the pod (generate + push dataset; blocks until pushed)"
  ssh "${ssh_opts[@]}" "root@${SSH_HOST}" \
    "HF_TOKEN='${HF_TOKEN}' REPO='${POD_REPO}' \
DATASET_REPO_ID='${DATASET_REPO_ID:-}' DATAGEN_ARGS='${DATAGEN_ARGS}' \
bash /root/datagen.sh"
fi

echo ">> running pod_setup.sh on the pod (install training stack + launch training in tmux)"
ssh "${ssh_opts[@]}" "root@${SSH_HOST}" \
  "HF_TOKEN='${HF_TOKEN}' WANDB_API_KEY='${WANDB_API_KEY:-}' \
DATASET_REPO_ID='${DATASET_REPO_ID:-}' POLICY_REPO_ID='${POLICY_REPO_ID:-}' \
STEPS='${STEPS:-}' SAVE_FREQ='${SAVE_FREQ:-}' BATCH_SIZE='${BATCH_SIZE:-}' \
JOB_NAME='${JOB_NAME:-}' OUTPUT_DIR='${OUTPUT_DIR:-}' METRICS_DIR='${METRICS_DIR:-}' \
DATASET_ROOT='${DATASET_ROOT:-}' HUB_PUSH='${HUB_PUSH:-}' RESUME='${RESUME:-}' \
POLICY_PATH='${POLICY_PATH:-}' LAUNCH='${LAUNCH:-train}' POD_REPO='${POD_REPO}' \
bash /root/pod_setup.sh"
