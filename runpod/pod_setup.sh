#!/usr/bin/env bash
set -euo pipefail

# Pod-side one-shot setup: build a clean Python 3.12 env with CUDA torch and
# lerobot[smolvla], authenticate `hf`, then launch training in a tmux session — or, with
# LAUNCH=dagger, a whole DAgger round (runpod/dagger_round.sh from the repo synced to POD_REPO), or
# with LAUNCH=baseline, a policy's baseline (runpod/baseline.sh, likewise), or with LAUNCH=dsrl, a
# DSRL steering run (runpod/dsrl.sh, likewise).
# Needs HF_TOKEN unless training reads a dataset already on disk (DATASET_ROOT) and pushes nothing
# (HUB_PUSH=false); WANDB_API_KEY is optional.

VENV="${VENV:-/root/vla-venv}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
TMUX_SESSION="${TMUX_SESSION:-train}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${HF_TOKEN:-}" && ( -z "${DATASET_ROOT:-}" || "${HUB_PUSH:-true}" == "true" ) ]]; then
  echo "FATAL: HF_TOKEN must be set to pull a private dataset or push checkpoints." >&2
  exit 1
fi

echo ">> apt deps (git, tmux, ffmpeg for video decode)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git tmux ffmpeg curl >/dev/null

echo ">> install uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

echo ">> create py3.12 venv at ${VENV} (kept if it exists, so a second launch reuses it)"
uv venv --python 3.12 --allow-existing "${VENV}"

echo ">> install CUDA torch stack from ${CUDA_INDEX}"
uv pip install --python "${VENV}/bin/python" \
  --index-url "${CUDA_INDEX}" torch torchvision torchcodec

# Pin lerobot to the exact version the datagen/eval pixi envs are solved against (pyproject.toml):
# the dataset written locally carries lerobot 0.6.1's v3.0 layout, so training must read it with
# the same version or `get_safe_version` / the parquet schema can drift out from under us.
LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"
echo ">> install lerobot[smolvla,dataset]==${LEROBOT_VERSION}"
uv pip install --python "${VENV}/bin/python" "lerobot[smolvla,dataset]==${LEROBOT_VERSION}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo ">> hf login (token from env)"
  "${VENV}/bin/hf" auth login --token "${HF_TOKEN}" --add-to-git-credential
fi

if [[ -n "${DATASET_ROOT:-}" && ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "FATAL: DATASET_ROOT=${DATASET_ROOT} holds no LeRobot dataset (meta/info.json missing)." >&2
  exit 1
fi

echo ">> sanity: import torch + cuda"
"${VENV}/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible'; print('cuda ok:', torch.cuda.get_device_name(0))"

if [[ "${LAUNCH:-train}" == "dagger" ]]; then
  POD_REPO="${POD_REPO:-/workspace/vla_chess}"
  echo ">> launch the DAgger round in tmux session '${TMUX_SESSION}'"
  tmux new-session -d -s "${TMUX_SESSION}" \
    "VENV='${VENV}' bash '${POD_REPO}/runpod/dagger_round.sh' 2>&1 | tee /root/dagger.log"
  echo ">> round started. Attach with:  tmux attach -t ${TMUX_SESSION}"
  echo ">> tail the log with:           tail -f /root/dagger.log"
  exit 0
fi

if [[ "${LAUNCH:-train}" == "baseline" ]]; then
  POD_REPO="${POD_REPO:-/workspace/vla_chess}"
  echo ">> launch the baseline in tmux session '${TMUX_SESSION}'"
  tmux new-session -d -s "${TMUX_SESSION}" \
    "bash '${POD_REPO}/runpod/baseline.sh' 2>&1 | tee /root/baseline.log"
  echo ">> baseline started. Attach with:  tmux attach -t ${TMUX_SESSION}"
  echo ">> tail the log with:              tail -f /root/baseline.log"
  exit 0
fi

if [[ "${LAUNCH:-train}" == "dsrl" ]]; then
  POD_REPO="${POD_REPO:-/workspace/vla_chess}"
  echo ">> launch the DSRL run in tmux session '${TMUX_SESSION}'"
  tmux new-session -d -s "${TMUX_SESSION}" \
    "bash '${POD_REPO}/runpod/dsrl.sh' 2>&1 | tee /root/dsrl.log"
  echo ">> run started. Attach with:  tmux attach -t ${TMUX_SESSION}"
  echo ">> tail the log with:         tail -f /root/dsrl.log"
  exit 0
fi

echo ">> launch training in tmux session '${TMUX_SESSION}'"
chmod +x "${HERE}/train.sh"
VENV="${VENV}" tmux new-session -d -s "${TMUX_SESSION}" \
  "WANDB_API_KEY='${WANDB_API_KEY:-}' VENV='${VENV}' \
DATASET_REPO_ID='${DATASET_REPO_ID:-}' POLICY_REPO_ID='${POLICY_REPO_ID:-}' \
STEPS='${STEPS:-}' SAVE_FREQ='${SAVE_FREQ:-}' BATCH_SIZE='${BATCH_SIZE:-}' \
JOB_NAME='${JOB_NAME:-}' OUTPUT_DIR='${OUTPUT_DIR:-}' METRICS_DIR='${METRICS_DIR:-}' \
DATASET_ROOT='${DATASET_ROOT:-}' HUB_PUSH='${HUB_PUSH:-}' RESUME='${RESUME:-}' \
POLICY_PATH='${POLICY_PATH:-}' \
bash '${HERE}/train.sh' 2>&1 | tee /root/train.log"

echo ">> training started. Attach with:  tmux attach -t ${TMUX_SESSION}"
echo ">> tail the log with:              tail -f /root/train.log"
