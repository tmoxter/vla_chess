# runpod/ — SmolVLA datagen + training on a rented GPU

Scripts to spin up a RunPod GPU pod, install the stack, generate the dataset on
the pod (GPU/EGL render), run the SmolVLA fine-tune, and monitor it. Checkpoints
are pushed to the Hub, so no persistent volume is needed and interruptions are
recoverable.

Pieces:

- `launch.py` — create / status / terminate the pod, list GPUs.
- `bootstrap.sh` — local driver: copies the pod scripts over, optionally syncs
  the repo and runs datagen, then launches training.
- `datagen.sh` — pod-side: builds an EGL env and runs `generate-dataset`, pushing
  the dataset to the Hub with its `v3.0` version tag.
- `pod_setup.sh` + `train.sh` — pod-side: build the training env and run
  `lerobot-train`, checkpointing to the Hub.
- `smoke.env` / `exp2.env` — per-run knobs sourced by `bootstrap.sh`.
- `render_parity.py` — EGL↔osmesa render parity probe (run before trusting a big
  datagen run).

## Smoke test first — prove the plumbing

Before the real Exp-2 harvest burns GPU hours, run the smoke test: a tiny
fingers-down vertical slice that exercises every seam end to end (EGL datagen →
LeRobot v3.0 serialize → Hub push + `v3.0` tag → training pulls it → single-cam
`rename_map` → checkpoint push → `/workspace` disk). One command after `create`:

```
DATAGEN=1 ENV_FILE=runpod/smoke.env ./runpod/bootstrap.sh <ssh_host> <ssh_port>
```

Success = the dataset repo (`chess-smoke-v1`) gains a `v3.0` tag **and** a
checkpoint appears under the policy repo (`smolvla-chess-smoke`). Then terminate
the pod and launch the real run with `ENV_FILE=runpod/exp2.env`.

## Secrets

Put credentials in `runpod/.env` (gitignored — never commit or paste in chat):

```
RUNPOD_API_KEY=...        # RunPod > Settings > API Keys
HF_TOKEN=...              # HF write token (private dataset pull + checkpoint push)
WANDB_API_KEY=...         # optional; omit to train with W&B disabled
```

## One-time RunPod account setup

1. Create the account, add credit.
2. Add the SSH **public** key to RunPod > Settings > SSH Public Keys:
   ```
   ~/.ssh/runpod_ed25519.pub
   ```
3. Create an API key, drop it (and the HF token) into `runpod/.env`.

## Local prerequisites

The control scripts need the RunPod SDK. Install it into the datagen env:

```
.pixi/envs/datagen/bin/pip install runpod
```

`watch.py` also uses `huggingface_hub`, already present in that env.

## Flow

```
# 1. pick a GPU (verify id + price — the catalogue changes)
python runpod/launch.py list-gpus

# 2. create the pod (defaults: RTX 4090, Community Cloud, 40 GB disk)
python runpod/launch.py create
#    -> prints {pod_id, ssh_host, ssh_port} and the bootstrap command

# 3. generate+push the dataset AND launch training (one command; smoke or exp2 env)
DATAGEN=1 ENV_FILE=runpod/exp2.env ./runpod/bootstrap.sh <ssh_host> <ssh_port>
#    - DATAGEN=1   rsyncs the repo and runs datagen.sh (generate + push) first.
#    - omit DATAGEN to train against an already-pushed dataset.
#    - omit ENV_FILE to use train.sh's Exp-1 overfit defaults (unchanged).

# 4. monitor (drive from cron for hands-off watching)
python runpod/watch.py <pod_id>

# 5. tear down when done
python runpod/launch.py terminate <pod_id>
```

## Datagen on the pod (GPU/EGL render)

`datagen.sh` runs `generate-dataset` under `MUJOCO_GL=egl` so rendering uses the
GPU (~2.6 s/frame → ~ms), which is what makes 30 fps affordable. It first renders
the parity scene to `/workspace/egl-parity.npy` — compare it against a local
osmesa render (`pixi run python runpod/render_parity.py ...`) before trusting a
big run. The flags come from `DATAGEN_ARGS` in the env file, and the dataset is
pushed to `--repo-id` with its `v3.0` tag via `--push` (needs `HF_TOKEN`).

To run datagen alone (no training) on a pod, after `bootstrap.sh` has synced the
repo:

```
HF_TOKEN=... DATASET_REPO_ID=... DATAGEN_ARGS='--funnel ... --push' \
  bash /root/datagen.sh      # on the pod
```

## Monitoring via cron

`watch.py` prints a JSON status line and exits with a coarse state code
(0 running, 2 checkpoint present, 3 pod gone, 1 error). A cron/heartbeat wrapper
runs it every few minutes, pings on milestones or failure, and terminates the pod
once the final checkpoint is on the Hub. Set that up once a pod is live.

## Recovery

If the pod dies, create a fresh one, bootstrap it, and resume from the Hub:

```
RESUME=true bash runpod/train.sh   # inside the pod (via bootstrap)
```

`RESUME=true` pulls the latest checkpoint (weights + optimizer + step + RNG) from
`hugginggeorge/smolvla-chess-overfit` and continues exactly.

## Note on the RunPod SDK

`launch.py`/`watch.py` target the `runpod` Python SDK (`create_pod`, `get_pod`,
`terminate_pod`, `get_gpus`). RunPod occasionally changes field names in the pod
runtime payload; if `create` can't find the SSH port, run `launch.py status
<pod_id>` and check the `runtime.ports` shape against what `_wait_for_ssh`
expects. Verified logic, unverified against a live account — sanity-check on first
real run.
