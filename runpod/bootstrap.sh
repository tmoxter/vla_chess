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
#
# End-to-end plumbing smoke test (datagen -> push -> train -> checkpoint push), one command:
#   DATAGEN=1 ENV_FILE=runpod/smoke.env ./runpod/bootstrap.sh <ssh_host> <ssh_port>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/runpod_ed25519}"
POD_REPO="${POD_REPO:-/root/vla_chess}"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <ssh_host> <ssh_port>" >&2
  exit 1
fi
SSH_HOST="$1"
SSH_PORT="$2"

if [[ -f "${HERE}/.env" ]]; then
  set -a; source "${HERE}/.env"; set +a
fi
# Run-specific knobs (dataset/policy repos, batch size, DATAGEN_ARGS, ...). Sourced after .env so a
# run file can reference secrets, and exported so they forward to the pod-side scripts below.
if [[ -n "${ENV_FILE:-}" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "FATAL: ENV_FILE ${ENV_FILE} not found." >&2
    exit 1
  fi
  set -a; source "${ENV_FILE}"; set +a
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN not set (put it in runpod/.env)." >&2
  exit 1
fi

ssh_opts=(-i "${SSH_KEY}" -p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new)

echo ">> copying pod scripts"
scp "${ssh_opts[@]/-p/-P}" \
  "${HERE}/pod_setup.sh" "${HERE}/train.sh" "${HERE}/datagen.sh" \
  "root@${SSH_HOST}:/root/"

if [[ "${DATAGEN:-0}" == "1" ]]; then
  if [[ -z "${DATAGEN_ARGS:-}" ]]; then
    echo "FATAL: DATAGEN=1 but DATAGEN_ARGS is empty (set ENV_FILE to smoke.env/exp2.env)." >&2
    exit 1
  fi
  echo ">> ensuring rsync is present on the pod"
  ssh "${ssh_opts[@]}" "root@${SSH_HOST}" \
    "command -v rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)"

  echo ">> rsyncing source tree to ${POD_REPO} on the pod"
  rsync -az --delete \
    --exclude '.git/' --exclude '.pixi/' --exclude '*-captures/' \
    --exclude 'data/' --exclude 'outputs/' --exclude 'mlflow/' \
    --exclude '__pycache__/' --exclude '.*_cache/' --exclude 'runpod/artifacts/' \
    -e "ssh ${ssh_opts[*]}" \
    "${REPO_ROOT}/" "root@${SSH_HOST}:${POD_REPO}/"

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
bash /root/pod_setup.sh"
