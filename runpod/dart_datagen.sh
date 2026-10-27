#!/usr/bin/env bash
set -euo pipefail

# DART datagen: the Exp-1 funnel recipe (one-step pushes, single task, 6 seed slots, 30 fps,
# FIRST selection, nominal board) with the kept episode recorded under action noise on every leg,
# the expert holding each leg (jaws included) until the arm is back on its command.
# Shards the 120 slots over SHARDS processes, then serializes once into a local LeRobot dataset.
# Prints EXIT_<code> at the end for polling.
#
# Runs on a GPU pod (Linux, EGL, the pod's venv) and on macOS (CGL, the pixi datagen env); every
# default below is picked per host and can be overridden from the environment.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" == "Darwin" ]]; then
  DEFAULT_GL=cgl
  DEFAULT_SHARDS="$(sysctl -n hw.ncpu)"
else
  DEFAULT_GL=egl
  DEFAULT_SHARDS=14
fi
if [[ -x /root/eval-venv/bin/python ]]; then
  DEFAULT_PYTHON=/root/eval-venv/bin/python
else
  DEFAULT_PYTHON="${REPO_DIR}/.pixi/envs/datagen/bin/python"
fi
if [[ -d /workspace/data ]]; then
  DEFAULT_DATA_ROOT=/workspace/data
else
  DEFAULT_DATA_ROOT="${HOME}/data/vla_chess"
fi

PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
SHARDS="${SHARDS:-${DEFAULT_SHARDS}}"
SIGMA="${SIGMA:-0.06}"
OUT="${OUT:-${DEFAULT_DATA_ROOT}/chess_pawns_funnel_dart_v2}"
CAPTURES="${OUT}-captures"
REPO_ID="${REPO_ID:-hugginggeorge/chess-pawns-funnel-dart-v2}"
LOGDIR="${LOGDIR:-${OUT}-logs}"
export MUJOCO_GL="${MUJOCO_GL:-${DEFAULT_GL}}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"

cd "${REPO_DIR}"
mkdir -p "${LOGDIR}" "${CAPTURES}"
echo "python=${PYTHON} MUJOCO_GL=${MUJOCO_GL} shards=${SHARDS} sigma=${SIGMA} out=${OUT}"
COMMON=(--funnel --one-step-only --single-task --seeds 6 --fps 30 --dart-sigma "${SIGMA}"
        --out "${OUT}" --captures "${CAPTURES}" --repo-id "${REPO_ID}")

pids=()
for ((i = 0; i < SHARDS; i++)); do
  "${PYTHON}" -m vla_chess.cli.generate_dataset "${COMMON[@]}" \
    --shard "${i}/${SHARDS}" --no-serialize >"${LOGDIR}/shard${i}.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
echo "shards done (fail=${fail}); kept: $(ls -d "${CAPTURES}"/ep-* 2>/dev/null | wc -l)/120"
[[ "${fail}" == 0 ]] || { echo "EXIT_1"; exit 1; }

# Every slot is captured now, so this pass only skips and serializes.
"${PYTHON}" -m vla_chess.cli.generate_dataset "${COMMON[@]}" 2>&1 | tail -n 3
echo "EXIT_0"
