#!/usr/bin/env bash
set -euo pipefail

# Pod-side one-shot setup for the eval-metrics sweep: a Python 3.12 venv with CUDA torch,
# lerobot[smolvla,dataset]==0.6.1, and the mujoco/mink scene stack the oracle + rollout need.
# Logs to /root/eval_setup.log and prints EXIT_<code> at the end for polling.

VENV="${VENV:-/root/eval-venv}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
REPO="${REPO:-/workspace/vla_chess}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"

echo ">> apt deps (ffmpeg, libgl/egl for mujoco EGL render)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 libegl1 curl >/dev/null

echo ">> install uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

echo ">> create py3.12 venv at ${VENV}"
uv venv --python 3.12 "${VENV}"

echo ">> CUDA torch stack from ${CUDA_INDEX}"
uv pip install --python "${VENV}/bin/python" \
  --index-url "${CUDA_INDEX}" torch torchvision torchcodec

echo ">> lerobot[smolvla,dataset]==${LEROBOT_VERSION} + scene stack (mujoco, mink), numpy<2.3"
uv pip install --python "${VENV}/bin/python" "lerobot[smolvla,dataset]==${LEROBOT_VERSION}"
uv pip install --python "${VENV}/bin/python" "mujoco>=3.2" "mink>=0.0.4" "numpy<2.3"

echo ">> editable install of vla_chess (waits for rsync to have landed pyproject+src)"
until [[ -f "${REPO}/pyproject.toml" && -d "${REPO}/src/vla_chess" ]]; do
  echo "   waiting for ${REPO} to be rsynced..."; sleep 5;
done
uv pip install --python "${VENV}/bin/python" --no-deps -e "${REPO}"

echo ">> sanity: torch CUDA + vla_chess import"
"${VENV}/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('cuda ok:', torch.cuda.get_device_name(0))"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${VENV}/bin/python" -c "import mujoco, mink, vla_chess.cli.eval_metrics; print('imports ok')"

echo "EXIT_0"
