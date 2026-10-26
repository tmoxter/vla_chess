#!/usr/bin/env bash
set -euo pipefail

# Pod-side training entrypoint. Assumes pod_setup.sh has created the venv at
# $VENV and logged `hf` in. Override any knob via environment variables.

VENV="${VENV:-/root/vla-venv}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
POLICY_REPO_ID="${POLICY_REPO_ID:-hugginggeorge/smolvla-chess-overfit}"
DATASET_REPO_ID="${DATASET_REPO_ID:-hugginggeorge/chess-pawns-v1}"
# SmolVLA's config declares camera1/2/3; our dataset has a single wrist cam.
# Renaming it to camera1 makes the provided cameras a subset of the policy's
# (the extra views are padded), which passes lerobot's feature-consistency check.
RENAME_MAP="${RENAME_MAP:-{\"observation.images.wrist\": \"observation.images.camera1\"}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/outputs/smolvla_chess_overfit}"
JOB_NAME="${JOB_NAME:-smolvla_chess_overfit}"
RESUME="${RESUME:-false}"
# Train from a dataset already on this pod's disk instead of pulling it from the Hub (the DART
# dataset is generated and trained on the same pod and never pushed). Empty = pull by repo id.
DATASET_ROOT="${DATASET_ROOT:-}"
# Push checkpoints and the final policy to the Hub. Off when Hub storage is full: Exp 2's run
# crashed mid-training on a checkpoint push and again on the end-of-run push.
HUB_PUSH="${HUB_PUSH:-true}"
# Where the durable training log lives. Point this at the persistent /workspace volume (exp2.env
# does) so a pod death does not take the loss history with it; it also survives for the MLflow
# ingest to replay. Defaults to a sibling of the checkpoint dir.
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}_logs}"
# lerobot-train's validate() aborts if OUTPUT_DIR already exists and resume is false. The durable
# log must therefore live OUTSIDE OUTPUT_DIR: if METRICS_DIR is OUTPUT_DIR (or a child of it),
# creating it pre-creates OUTPUT_DIR and the run fails its own existence check. Redirect to a sibling.
if [[ "${RESUME}" != "true" && ( "${METRICS_DIR}" == "${OUTPUT_DIR}" || "${METRICS_DIR}" == "${OUTPUT_DIR}/"* ) ]]; then
  echo "WARN: METRICS_DIR == OUTPUT_DIR would trip lerobot's exists-check; using ${OUTPUT_DIR}_logs instead." >&2
  METRICS_DIR="${OUTPUT_DIR}_logs"
fi
mkdir -p "${METRICS_DIR}"

wandb_args=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb_args=(--wandb.enable=true --wandb.project="${WANDB_PROJECT:-vla_chess}")
else
  echo "WARN: WANDB_API_KEY unset -> running with W&B disabled (no loss curves)." >&2
  wandb_args=(--wandb.enable=false)
fi

dataset_args=(--dataset.repo_id="${DATASET_REPO_ID}")
if [[ -n "${DATASET_ROOT}" ]]; then
  dataset_args+=(--dataset.root="${DATASET_ROOT}")
fi

resume_args=()
if [[ "${RESUME}" == "true" ]]; then
  # Resume pulls the latest checkpoint from the Hub policy repo.
  resume_args=(--resume=true --config_path="${POLICY_REPO_ID}")
else
  resume_args=(--policy.path="${POLICY_PATH}")
fi

# Pipe through tee (not exec) so the full console log lands on the durable volume as well as the
# tmux pipe. lerobot-train has no native JSONL metrics hook (its only logger is W&B, and the
# console line abbreviates step counts), so this log is the durable record; a structured
# metrics.jsonl for MLflow is a separate shim, not pretended here.
set -o pipefail
"${VENV}/bin/lerobot-train" \
  "${resume_args[@]}" \
  --policy.repo_id="${POLICY_REPO_ID}" --policy.private=true --policy.push_to_hub="${HUB_PUSH}" \
  "${dataset_args[@]}" \
  --rename_map="${RENAME_MAP}" \
  --batch_size="${BATCH_SIZE}" --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" --save_checkpoint=true --save_checkpoint_to_hub="${HUB_PUSH}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  "${wandb_args[@]}" 2>&1 | tee -a "${METRICS_DIR}/train.log"
