"""Create, list, and terminate the RunPod GPU pod for SmolVLA training.

Thin wrapper over the RunPod Python SDK. Reads RUNPOD_API_KEY from the
environment (or runpod/.env). Verify GPU ids/prices with `--list-gpus` before
creating, since RunPod's catalogue and pricing change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import runpod  # type: ignore[import-untyped]

DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_NAME = "vla-chess-smolvla"


def _load_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _require_api_key() -> None:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("FATAL: RUNPOD_API_KEY not set (put it in runpod/.env).", file=sys.stderr)
        sys.exit(1)
    runpod.api_key = key


def list_gpus() -> None:
    for gpu in runpod.get_gpus():
        print(f"{gpu['id']:<34} {gpu.get('memoryInGb', '?')} GB")


def create(args: argparse.Namespace) -> None:
    pod = runpod.create_pod(
        name=args.name,
        image_name=args.image,
        gpu_type_id=args.gpu,
        cloud_type=args.cloud,
        gpu_count=1,
        container_disk_in_gb=args.disk,
        volume_in_gb=0,
        ports="22/tcp",
        support_public_ip=True,
        start_ssh=True,
    )
    pod_id = pod["id"]
    print(f"created pod {pod_id}; waiting for SSH endpoint...")
    host, port = _wait_for_ssh(pod_id, timeout_s=args.timeout)
    print(json.dumps({"pod_id": pod_id, "ssh_host": host, "ssh_port": port}, indent=2))
    print(f"\nnext:  ./bootstrap.sh {host} {port}")


def _wait_for_ssh(pod_id: str, timeout_s: int) -> tuple[str, int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        runtime = pod.get("runtime") or {}
        for p in runtime.get("ports") or []:
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                return p["ip"], int(p["publicPort"])
        time.sleep(5)
    raise TimeoutError(f"pod {pod_id} did not expose a public SSH port in {timeout_s}s")


def status(args: argparse.Namespace) -> None:
    pod = runpod.get_pod(args.pod_id)
    runtime = pod.get("runtime") or {}
    print(
        json.dumps(
            {
                "id": pod["id"],
                "desiredStatus": pod.get("desiredStatus"),
                "uptimeSeconds": runtime.get("uptimeInSeconds"),
                "gpu": pod.get("machine", {}).get("gpuDisplayName"),
            },
            indent=2,
        )
    )


def terminate(args: argparse.Namespace) -> None:
    runpod.terminate_pod(args.pod_id)
    print(f"terminated {args.pod_id}")


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-gpus", help="print available GPU type ids")

    c = sub.add_parser("create", help="create the training pod")
    c.add_argument("--gpu", default=DEFAULT_GPU)
    c.add_argument("--image", default=DEFAULT_IMAGE)
    c.add_argument("--name", default=DEFAULT_NAME)
    c.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE", "ALL"])
    c.add_argument("--disk", type=int, default=40, help="container disk GB")
    c.add_argument("--timeout", type=int, default=300, help="seconds to wait for SSH")
    c.set_defaults(func=create)

    s = sub.add_parser("status", help="print pod status")
    s.add_argument("pod_id")
    s.set_defaults(func=status)

    t = sub.add_parser("terminate", help="terminate a pod")
    t.add_argument("pod_id")
    t.set_defaults(func=terminate)

    args = parser.parse_args()
    _require_api_key()
    if args.cmd == "list-gpus":
        list_gpus()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
