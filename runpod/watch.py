"""Report training progress for a RunPod pod, for cron-driven monitoring.

Prints one JSON status line combining pod liveness (RunPod SDK) and the latest
checkpoint step pushed to the Hub policy repo (huggingface_hub). Exit code
signals coarse state so a cron wrapper can decide whether to ping or terminate:
  0 running, 2 checkpoint(s) present, 3 pod gone/stopped, 1 error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import runpod  # type: ignore[import-untyped]
from huggingface_hub import HfApi


def _load_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def latest_checkpoint_step(repo_id: str, token: str | None) -> int | None:
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model", token=token)
    prefix = "checkpoints/"
    steps = {
        int(name)
        for f in files
        if f.startswith(prefix) and (name := f[len(prefix) :].split("/", 1)[0]).isdigit()
    }
    return max(steps) if steps else None


def main() -> int:
    _load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pod_id")
    parser.add_argument("--policy-repo", default="hugginggeorge/smolvla-chess-overfit")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("FATAL: RUNPOD_API_KEY not set.", file=sys.stderr)
        return 1
    runpod.api_key = api_key

    try:
        pod = runpod.get_pod(args.pod_id)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1

    runtime = pod.get("runtime") or {} if pod else {}
    desired = pod.get("desiredStatus") if pod else None
    step = latest_checkpoint_step(args.policy_repo, os.environ.get("HF_TOKEN"))

    print(
        json.dumps(
            {
                "pod_id": args.pod_id,
                "desiredStatus": desired,
                "uptimeSeconds": runtime.get("uptimeInSeconds"),
                "latestCheckpointStep": step,
            }
        )
    )

    if not pod or desired != "RUNNING":
        return 3
    if step is not None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
