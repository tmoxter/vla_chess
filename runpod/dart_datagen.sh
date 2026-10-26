#!/usr/bin/env bash
set -euo pipefail

# DART datagen: the Exp-1 funnel recipe (one-step pushes, single task, 6 seed slots, 30 fps,
# FIRST selection, nominal board) with the kept episode recorded under phase-gated action noise.
# Shards the 120 slots over SHARDS processes, then serializes once into a local LeRobot dataset.
# Prints EXIT_<code> at the end for polling.

VENV="${VENV:-/root/eval-venv}"
SHARDS="${SHARDS:-14}"
SIGMA="${SIGMA:-0.06}"
OUT="${OUT:-/workspace/data/chess_pawns_funnel_dart}"
CAPTURES="${OUT}-captures"
REPO_ID="${REPO_ID:-hugginggeorge/chess-pawns-funnel-dart-v1}"
LOGDIR="${LOGDIR:-/root/dart_datagen_logs}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

cd /workspace/vla_chess
mkdir -p "${LOGDIR}" "${CAPTURES}"
COMMON=(--funnel --one-step-only --single-task --seeds 6 --fps 30 --dart-sigma "${SIGMA}"
        --out "${OUT}" --captures "${CAPTURES}" --repo-id "${REPO_ID}")

pids=()
for ((i = 0; i < SHARDS; i++)); do
  "${VENV}/bin/python" -m vla_chess.cli.generate_dataset "${COMMON[@]}" \
    --shard "${i}/${SHARDS}" --no-serialize >"${LOGDIR}/shard${i}.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
echo "shards done (fail=${fail}); kept: $(ls -d "${CAPTURES}"/ep-* 2>/dev/null | wc -l)/120"
[[ "${fail}" == 0 ]] || { echo "EXIT_1"; exit 1; }

# Every slot is captured now, so this pass only skips and serializes.
"${VENV}/bin/python" -m vla_chess.cli.generate_dataset "${COMMON[@]}" 2>&1 | tail -n 3
echo "EXIT_0"
