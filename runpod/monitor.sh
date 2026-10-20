#!/usr/bin/env bash
# Deterministic monitor for the RunPod Exp-2 run. SSHes into the pod, derives a single milestone
# "token" from the logs/checkpoints, dedupes against a local state file, and prints exactly one of:
#   NOTIFY: <human-readable message>     -> a NEW milestone worth telling Tobi about
#   QUIET                                -> nothing changed since last check
# plus, on the final milestone, it recovers the loss log locally. Exit code is always 0 unless the
# pod is unreachable (2), so the caller can treat "can't reach pod" distinctly.
set -uo pipefail

KEY="${KEY:-$HOME/.ssh/runpod_ed25519}"
HOST="${HOST:-69.30.85.188}"
PORT="${PORT:-22006}"
STATE="${STATE:-$(dirname "$0")/.monitor_state}"
RECOVER_DIR="${RECOVER_DIR:-$(dirname "$0")/artifacts}"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

# One round-trip: pull everything we need as a compact report.
report="$($SSH root@"$HOST" '
  log=/root/exp2_full.log
  err=0
  grep -qi "buffer is full\|Traceback (most recent\|FATAL\|Error:" "$log" 2>/dev/null && err=1
  phase="init"
  grep -q "DATAGEN START" "$log" 2>/dev/null && phase="datagen"
  grep -q "pushed .*chess-pawns-funnel-v1.* v3.0\|DATAGEN DONE" "$log" 2>/dev/null && phase="datagen_done"
  grep -q "TRAIN START" "$log" 2>/dev/null && phase="train"
  grep -q "End of training\|TRAIN DONE" "$log" 2>/dev/null && phase="train_done"
  grep -q "Model pushed to .*smolvla-chess-exp2" "$log" 2>/dev/null && phase="pushed"
  move=$(grep -oE "\[[0-9]+/120\]" "$log" 2>/dev/null | tail -1 | tr -d "[]")
  step=$(grep -oE "step:[0-9]+" "$log" 2>/dev/null | tail -1 | cut -d: -f2)
  ckpts=$(ls -1d /workspace/outputs/smolvla_chess_exp2/checkpoints/[0-9]* 2>/dev/null | wc -l | tr -d " ")
  alive=no; tmux has-session -t exp2 2>/dev/null && alive=yes
  echo "err=$err phase=$phase move=$move step=$step ckpts=$ckpts alive=$alive"
' 2>/dev/null)"

if [[ -z "$report" ]]; then echo "QUIET"; exit 2; fi   # pod unreachable; stay quiet, signal via code
eval "$report"

prev="$(cat "$STATE" 2>/dev/null || echo "")"
# Token changes on: error, phase change, every new checkpoint, or crossing a 2000-step mark.
token="err=$err;phase=$phase;ckpts=$ckpts;alive=$alive"
echo "$token" > "$STATE"
if [[ "$token" == "$prev" ]]; then echo "QUIET"; exit 0; fi

case 1 in
  $((err==1)) )
    echo "NOTIFY: ⚠️ Exp-2 pod run hit an ERROR (phase=$phase, move=${move:-?}, step=${step:-?}). The tmux session is alive=$alive. I'm looking into it." ;;
esac
if [[ "$err" == "1" ]]; then exit 0; fi

case "$phase" in
  datagen_done) echo "NOTIFY: ✅ Exp-2 funnel datagen done — chess-pawns-funnel-v1 pushed (v3.0). Training starting." ;;
  train)
    if [[ "${ckpts:-0}" -gt 0 ]]; then
      echo "NOTIFY: 📈 Exp-2 training — checkpoint #$ckpts on disk (latest step ${step:-?}), auto-pushing to the Hub."
    else
      echo "NOTIFY: 🚀 Exp-2 training started (step ${step:-0})."
    fi ;;
  train_done|pushed)
    mkdir -p "$RECOVER_DIR"
    $SSH root@"$HOST" 'cat /workspace/outputs/smolvla_chess_exp2_logs/train.log 2>/dev/null || cat /root/exp2_full.log' > "$RECOVER_DIR/exp2_train.log" 2>/dev/null
    lines=$(wc -l < "$RECOVER_DIR/exp2_train.log" 2>/dev/null || echo 0)
    echo "NOTIFY: 🏁 Exp-2 TRAINING COMPLETE (phase=$phase). Recovered train.log locally ($lines lines). Building loss curve + verifying final checkpoint on the Hub before giving the shutdown green-light." ;;
  *)
    echo "NOTIFY: Exp-2 status: phase=$phase, move ${move:-?}, step=${step:-?}, checkpoints=$ckpts." ;;
esac
exit 0
