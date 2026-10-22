# Strategy — Step 2: SmolVLA supervised fine-tuning (overfit sanity run)

Target picture for the second phase: take the `chess-pawns-v1` dataset produced in
step 1 and fine-tune a pretrained SmolVLA checkpoint on it, on a rented GPU. The
first milestone is deliberately small — **overfit the 30 oracle episodes** to prove
the (image, state, instruction) -> action mapping is learnable end to end with our
data format, before spending on a larger dataset or a real eval harness.

## Goal of this phase

Produce a fine-tuned SmolVLA checkpoint that has demonstrably memorised the oracle
trajectories: training loss collapses toward zero and offline action error on the
training frames is tiny. This validates the data, the training wiring, and the
checkpoint-persistence story — nothing more.

## Why overfit first

- It isolates *data/plumbing* failures from *learning* failures. If SmolVLA cannot
  even memorise 30 clean scripted episodes, something is wrong with the dataset,
  the normalisation, or the action space — and that is far cheaper to find now.
- It is fast and cheap (a single consumer GPU, ~2 hours, well under $5).
- It gives a known-good checkpoint to build the sim-eval harness against in step 3.

## The model

- `lerobot/smolvla_base` — 450M params (SmolVLM2 backbone + action expert),
  designed to fine-tune on a single consumer GPU. Loaded via `--policy.path`.
- Fine-tune, do not train from scratch (`--policy.path`, not `--policy.type`).

## Dataset

- `hugginggeorge/chess-pawns-v1` (private on HF Hub), v3.0, 30 episodes / 4830
  frames / 4 tasks (centre-file pawn pushes c/d/e/f 7->6), single wrist camera at
  512x512 AV1, `observation.state` and `action` both 6-D (so101 joints + gripper).
- Training pulls it straight from the Hub (needs `hf auth login`).

## Compute

- **RunPod, RTX 4090 (24 GB), Community Cloud.** bf16 SmolVLA fits comfortably at
  batch 16; the overfit run is short. Prices move — verify live before booking.
- No persistent volume. Checkpoints are pushed to the Hub every `save_freq` steps
  (`--save_checkpoint_to_hub`), and resume pulls the full checkpoint (weights +
  optimizer + step + RNG) back, so an interrupted or dead pod loses nothing.
- This makes **spot/interruptible pods safe**: an interruption just resumes from
  the last pushed checkpoint.

## Persistence — verified against lerobot 0.6.1 source

- `save_checkpoint(dir)` writes `checkpoints/<step>/{pretrained_model,training_state}/`.
- `push_checkpoint_to_hub(dir, repo_id)` uploads the **entire** step dir (weights
  *and* training_state) to `checkpoints/<step>/` on the Hub, tagged with the step.
- `--resume=true --config_path=<hub repo id>` calls `find_latest_hub_checkpoint`,
  downloads the highest-step dir, and continues from exact optimizer/step/RNG state.
- Recovery drill: pod dies -> new pod -> `--resume=true
  --config_path=hugginggeorge/smolvla-chess-overfit`.

## Training command

See `runpod/train.sh`. Canonical form:

```
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=hugginggeorge/smolvla-chess-overfit --policy.private=true \
  --dataset.repo_id=hugginggeorge/chess-pawns-v1 \
  --batch_size=16 --steps=20000 \
  --save_freq=2000 --save_checkpoint=true --save_checkpoint_to_hub=true \
  --output_dir=/root/outputs/smolvla_chess_overfit \
  --job_name=smolvla_chess_overfit \
  --wandb.enable=true --wandb.project=vla_chess
```

## Success criteria

- Training loss trends toward ~0 over the run (watch in W&B).
- A reload of the final checkpoint reproduces the oracle actions on training frames
  to low MSE (offline, no sim needed).

## Explicitly out of scope for this phase

- **Rollout eval in sim** — needs the MuJoCo scene wrapped as a LeRobot/gym env,
  which does not exist yet. That is step 3, and it is what turns "loss collapsed"
  into "the policy actually plays the move". Until then, eval is offline only.
- Generalisation, held-out tasks, real hardware, RL fine-tuning.

## Operational runbook

The `runpod/` directory holds the scripts that drive this:

1. `runpod/launch.py` — list GPUs / create the pod / terminate it (RunPod SDK).
2. `runpod/bootstrap.sh` — scp the pod-side scripts to the pod and run setup.
3. `runpod/pod_setup.sh` — pod-side: py3.12 venv, CUDA torch, `lerobot[smolvla]`,
   `hf` login, then launch `train.sh` in tmux.
4. `runpod/train.sh` — the `lerobot-train` invocation above.
5. `runpod/watch.py` — poll pod status + latest Hub checkpoint step; driven by a
   cron/heartbeat job that pings on progress, completion, or failure, and can
   auto-terminate the pod when done.

Secrets (`RUNPOD_API_KEY`, `HF_TOKEN`, optional `WANDB_API_KEY`) live in
`runpod/.env` (gitignored), never in the repo or chat.
