#!/usr/bin/env bash
set -euo pipefail

# Pod-side one-shot setup: build a clean Python 3.12 env with CUDA torch and
# lerobot[smolvla], authenticate `hf`, then launch training in a tmux session.
# Expects HF_TOKEN in the environment; WANDB_API_KEY is optional.

VENV="${VENV:-/root/vla-venv}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
TMUX_SESSION="${TMUX_SESSION:-train}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN must be set (needed for private dataset pull + checkpoint push)." >&2
  exit 1
fi

echo ">> apt deps (git, tmux, ffmpeg for video decode)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git tmux ffmpeg curl >/dev/null

echo ">> install uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

echo ">> create py3.12 venv at ${VENV}"
uv venv --python 3.12 "${VENV}"

echo ">> install CUDA torch stack from ${CUDA_INDEX}"
uv pip install --python "${VENV}/bin/python" \
  --index-url "${CUDA_INDEX}" torch torchvision torchcodec

# Pin lerobot to the exact version the datagen/eval pixi envs are solved against (pyproject.toml):
# the dataset written locally carries lerobot 0.6.1's v3.0 layout, so training must read it with
# the same version or `get_safe_version` / the parquet schema can drift out from under us.
LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"
echo ">> install lerobot[smolvla,dataset]==${LEROBOT_VERSION}"
uv pip install --python "${VENV}/bin/python" "lerobot[smolvla,dataset]==${LEROBOT_VERSION}"

echo ">> hf login (token from env)"
"${VENV}/bin/hf" auth login --token "${HF_TOKEN}" --add-to-git-credential

echo ">> sanity: import torch + cuda"
"${VENV}/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible'; print('cuda ok:', torch.cuda.get_device_name(0))"

echo ">> launch training in tmux session '${TMUX_SESSION}'"
chmod +x "${HERE}/train.sh"
VENV="${VENV}" tmux new-session -d -s "${TMUX_SESSION}" \
  "WANDB_API_KEY='${WANDB_API_KEY:-}' VENV='${VENV}' \
DATASET_REPO_ID='${DATASET_REPO_ID:-}' POLICY_REPO_ID='${POLICY_REPO_ID:-}' \
STEPS='${STEPS:-}' SAVE_FREQ='${SAVE_FREQ:-}' BATCH_SIZE='${BATCH_SIZE:-}' \
JOB_NAME='${JOB_NAME:-}' OUTPUT_DIR='${OUTPUT_DIR:-}' METRICS_DIR='${METRICS_DIR:-}' \
bash '${HERE}/train.sh' 2>&1 | tee /root/train.log"

echo ">> training started. Attach with:  tmux attach -t ${TMUX_SESSION}"
echo ">> tail the log with:              tail -f /root/train.log"
