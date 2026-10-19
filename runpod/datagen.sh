#!/usr/bin/env bash
set -euo pipefail

# Pod-side datagen: build an EGL-capable Python env and generate + push a LeRobot dataset using
# the GPU for rendering. MuJoCo physics is CPU either way; only the camera render moves to the GPU
# (MUJOCO_GL=egl), which is the ~2.6 s/frame -> ~ms speedup that makes 30 fps affordable. The
# finished dataset is pushed to the Hub with its v3.0 version tag so training can resolve it.
#
# Expects the vla_chess source already synced to $REPO (bootstrap.sh rsyncs it) and HF_TOKEN in the
# environment. The generate-dataset flags come in through $DATAGEN_ARGS so one script drives both
# the smoke slice and the real funnel harvest. Run directly on the pod, or via bootstrap.sh.

REPO="${REPO:-/root/vla_chess}"
DATAGEN_VENV="${DATAGEN_VENV:-/root/datagen-venv}"
CPU_INDEX="${CPU_INDEX:-https://download.pytorch.org/whl/cpu}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN must be set (needed to push the dataset)." >&2
  exit 1
fi
if [[ -z "${DATAGEN_ARGS:-}" ]]; then
  echo "FATAL: DATAGEN_ARGS must be set (the generate-dataset flags, e.g. from smoke.env)." >&2
  exit 1
fi
if [[ ! -d "${REPO}" ]]; then
  echo "FATAL: no source tree at ${REPO}; bootstrap.sh should have synced it." >&2
  exit 1
fi

echo ">> apt deps (git, ffmpeg for AV1 encode, EGL loader + libs for headless GPU render)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git ffmpeg curl libglvnd0 libegl1 libgles2 libgl1 >/dev/null

echo ">> install uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

echo ">> create py3.12 datagen venv at ${DATAGEN_VENV}"
uv venv --python 3.12 "${DATAGEN_VENV}"

# CPU torch is enough for datagen: rendering goes through EGL (not torch), and lerobot only touches
# torch to pack the dataset. The CPU index keeps the env small and mirrors the local datagen pins.
echo ">> install CPU torch stack from ${CPU_INDEX}"
uv pip install --python "${DATAGEN_VENV}/bin/python" \
  --index-url "${CPU_INDEX}" torch torchvision torchcodec

echo ">> install lerobot[dataset]==${LEROBOT_VERSION}, mujoco, and the vla_chess package"
uv pip install --python "${DATAGEN_VENV}/bin/python" \
  "lerobot[dataset]==${LEROBOT_VERSION}" mujoco "numpy>=1.26,<2.3" "mink>=0.0.4"
uv pip install --python "${DATAGEN_VENV}/bin/python" --no-deps -e "${REPO}"

echo ">> hf login (token from env)"
"${DATAGEN_VENV}/bin/hf" auth login --token "${HF_TOKEN}" --add-to-git-credential

# The renderer reads MUJOCO_GL once, at first GL init, and refuses anything but osmesa/egl.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo ">> sanity: EGL render of the parity scene (proves headless GPU rendering works)"
"${DATAGEN_VENV}/bin/python" "${REPO}/runpod/render_parity.py" \
  render --out /workspace/egl-parity.npy
echo ">> (compare /workspace/egl-parity.npy against a local osmesa render before trusting a big run)"

echo ">> generate + push dataset: generate-dataset ${DATAGEN_ARGS}"
# shellcheck disable=SC2086  # DATAGEN_ARGS is intentionally word-split into flags.
"${DATAGEN_VENV}/bin/python" -m vla_chess.cli.generate_dataset ${DATAGEN_ARGS}

echo ">> datagen done."
