"""Closed-loop SmolVLA rollout in the sim: let a fine-tuned policy drive the arm and render it.

This is the offline validation the training strategy deferred to "step 3": reconstruct one
recorded episode's initial scene, then at the dataset frame rate feed the policy the same
observation the dataset carried (wrist image + 6-D joint/gripper state + the task string),
apply its predicted absolute joint target through the identical control path the oracle used,
and write the rendered frames to an mp4.

Runs in the `eval` pixi env (it imports lerobot's SmolVLA stack). Example:

    pixi run -e eval eval-rollout --episode 0 --out runpod/artifacts/rollout-ep0.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# The renderer refuses to import unless the software GL backend is selected; the pixi env sets
# this, but a bare `python -m` invocation might not, so make the module import-safe either way.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco

from vla_chess.cli.generate_dataset import (
    SINGLE_PAWN_INSTRUCTION,
    build_specs,
)
from vla_chess.control.executor import Executor, PlacementCheck
from vla_chess.control.planner import MoveSpec
from vla_chess.dataset.episode import WRIST_CAMERA, move_scene_description
from vla_chess.dataset.observe import DEFAULT_FPS
from vla_chess.sensor.render import Renderer
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    PieceType,
    build_scene,
    make_data,
)

# Control runs at 50 Hz with 10 sim substeps; the dataset was captured at 10 fps, so one policy
# step advances the sim by one frame interval = (50/10) control steps x 10 substeps.
CONTROL_STEPS_PER_FRAME = 5
SUBSTEPS_PER_CONTROL = 10
SUBSTEPS_PER_FRAME = CONTROL_STEPS_PER_FRAME * SUBSTEPS_PER_CONTROL


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_policy(repo_id: str, revision: str, device: str) -> tuple[Any, Any, Any]:
    """Load the fine-tuned SmolVLA checkpoint and its bundled pre/post processors on CPU."""
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(os.cpu_count() or 1)
    root = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=[f"checkpoints/{revision}/pretrained_model/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    mdir = os.path.join(root, "checkpoints", revision, "pretrained_model")
    cfg = SmolVLAConfig.from_pretrained(mdir)
    cfg.device = device
    cfg.use_amp = False
    policy = SmolVLAPolicy.from_pretrained(mdir, config=cfg)
    policy.eval()
    policy.to(device)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=mdir,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def _observation(frame: np.ndarray, state: np.ndarray, instruction: str) -> dict[str, object]:
    """Assemble one LeRobot-format observation: float CHW image in [0,1], state, task string.

    The dataset's single wrist camera was renamed to `camera1` at training time (SmolVLA declares
    camera1/2/3 and pads the missing views), so the policy expects the wrist frame under that key.
    """
    import torch

    img = torch.from_numpy(frame).to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    return {
        "observation.state": torch.from_numpy(state).to(torch.float32).unsqueeze(0),
        "observation.images.camera1": img,
        "task": [instruction],
    }


def _spec_for(episode: int, source: str | None, target: str | None) -> MoveSpec:
    """Build the move a rollout drives: an arbitrary square pair, or a replayed training episode.

    With `source`/`target` set, runs that move (e.g. a held-out start square never seen in
    training) built the same way as the dataset specs but with the location overridden; otherwise
    replays training episode `episode`.

    Args:
        episode: Index into the training move cycle, used when source/target are not given.
        source: Start square to override with, or None to replay the episode.
        target: Destination square to override with, or None to replay the episode.

    Returns:
        The move spec, carrying the single-task instruction the first fine-tune was trained on.
    """
    if source is not None and target is not None:
        return MoveSpec(
            source=source,
            target=target,
            piece_type=PieceType.PAWN,
            instruction_override=SINGLE_PAWN_INSTRUCTION,
        )
    return build_specs(episode + 1, instruction=SINGLE_PAWN_INSTRUCTION)[episode]


def drive(
    policy: Any,
    pre: Any,
    post: Any,
    spec: MoveSpec,
    max_frames: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, PlacementCheck]:
    """Let an already-loaded policy drive the arm for one move, and score where the piece ends.

    At the dataset frame rate the policy is fed the same observation the dataset carried (wrist
    image + 6-D state + task string), its predicted absolute joint target goes through the oracle's
    control path, and the sim is stepped one frame on. After the fixed number of frames the piece's
    resting pose is scored the same way the oracle's `run` scores a placement.

    Args:
        policy: The loaded SmolVLA policy; its action queue is reset here before driving.
        pre: The policy's preprocessor.
        post: The policy's postprocessor.
        spec: The move to drive, carrying its start/target squares and instruction.
        max_frames: How many frames to drive before scoring.

    Returns:
        The rendered wrist frames, the per-frame observed state (before its action), the per-frame
        predicted action, and the placement verdict (final square, settled, success).
    """
    import torch

    instruction = spec.instruction()
    _log(f"driving {spec.source}->{spec.target}  task={instruction!r}")

    pieces = single_pawn_position(spec.source)
    description = move_scene_description(spec, pieces, DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS)
    model, index = build_scene(description)
    data = make_data(model, index)
    renderer = Renderer(model, [WRIST_CAMERA])
    executor = Executor(model, index)
    policy.reset()

    frames: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for step in range(max_frames):
        frame = np.array(renderer.render(data)[WRIST_CAMERA.name], copy=True)
        frames.append(frame)
        state = executor._state(data)
        obs = _observation(frame, state, instruction)
        with torch.inference_mode():
            action = post(policy.select_action(pre(obs)))
        action_np = action.squeeze(0).to(torch.float32).cpu().numpy()
        states.append(state.astype(np.float32))
        actions.append(action_np)
        executor._write_ctrl(data, action_np[:5], float(action_np[5]))
        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)
        if step % 10 == 0:
            _log(f"  step {step}/{max_frames}  state={np.round(state, 3)}")
    check = executor.placement(data, spec.source, spec.target)
    verdict = "SUCCESS" if check.placed else "FAIL"
    _log(
        f"rollout done: {len(frames)} frames; {spec.source}->{spec.target} "
        f"ended on {check.final_square} (settled={check.settled}) -> {verdict}"
    )
    return frames, np.stack(states), np.stack(actions), check


def rollout(
    repo_id: str,
    revision: str,
    episode: int,
    max_frames: int,
    device: str,
    source: str | None = None,
    target: str | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, PlacementCheck]:
    """Load the policy and drive one move, scoring it. See `drive` for the per-step loop."""
    _log("loading policy (first inference is slow on CPU)...")
    policy, pre, post = _load_policy(repo_id, revision, device)
    return drive(policy, pre, post, _spec_for(episode, source, target), max_frames)


def analyze_jitter(actions: np.ndarray, states: np.ndarray, chunk: int) -> None:
    """Print whether motion discontinuities concentrate at action-chunk boundaries.

    Compares the step-to-step change at chunk boundaries (steps chunk, 2*chunk, ...) against the
    interior steps, for both the commanded action and the realized state (5 arm joints, radians).
    """
    for label, seq in (("commanded action", actions), ("realized state", states)):
        delta = np.linalg.norm(np.diff(seq[:, :5], axis=0), axis=1)  # per-step arm-joint L2, rad
        idx = np.arange(1, len(seq))  # delta[i-1] is the change into step i
        boundary_mask = (idx % chunk) == 0
        boundary = delta[boundary_mask[: len(delta)]]
        interior = delta[~boundary_mask[: len(delta)]]
        _log(f"--- jitter: {label} (arm-joint step delta, rad) ---")
        _log(
            f"    interior: median={np.median(interior):.4f} mean={interior.mean():.4f} "
            f"p95={np.percentile(interior, 95):.4f} max={interior.max():.4f}"
        )
        _log(
            f"    boundary (steps {list(idx[boundary_mask[: len(delta)]])}): "
            f"vals={np.round(boundary, 4).tolist()} mean={boundary.mean():.4f}"
        )
        ratio = boundary.mean() / max(np.median(interior), 1e-9)
        _log(f"    boundary/interior-median ratio = {ratio:.1f}x")
        worst = int(np.argmax(delta)) + 1
        _log(f"    largest single jump at step {worst} (delta={delta[worst - 1]:.4f})")


def write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> bool:
    """Encode RGB frames to an mp4 via ffmpeg (mirrors cli.generate_episode.write_video)."""
    if not frames:
        return False
    height, width, _ = frames[0].shape
    path.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg ships in the pixi env's bin; fall back to it when PATH is bare (e.g. the env's python
    # was invoked directly rather than through `pixi run`).
    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.prefix) / "bin" / "ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    result = subprocess.run(
        command, input=b"".join(f.tobytes() for f in frames), capture_output=True, check=False
    )
    if result.returncode != 0:
        _log(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-500:]}")
        return False
    return True


def parse_moves(spec: str) -> list[tuple[str, str]]:
    """Parse a batch move list like "g7:g6,d6:d5" into (source, target) pairs.

    Args:
        spec: Comma-separated `source:target` square pairs.

    Returns:
        The parsed pairs, in order.

    Raises:
        ValueError: If any item is not a single `source:target` pair.
    """
    moves: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        source, sep, target = item.partition(":")
        if not sep or not source or not target:
            raise ValueError(f"move {item!r} is not a 'source:target' pair, e.g. g7:g6")
        moves.append((source, target))
    if not moves:
        raise ValueError(f"no moves parsed from {spec!r}")
    return moves


def _artifact_path(out: Path, source: str, target: str) -> Path:
    """Name one move's output file inside a batch so moves do not overwrite each other."""
    return out.with_name(f"{out.stem}-{source}{target}{out.suffix}")


def _finish_move(
    frames: list[np.ndarray], states: np.ndarray, actions: np.ndarray, out: Path
) -> None:
    """Persist traces, run the jitter analysis, and encode the mp4 for one driven move.

    Args:
        frames: Rendered wrist frames to encode.
        states: Per-frame observed state trace.
        actions: Per-frame predicted action trace.
        out: Where the mp4 goes; the trace npz sits beside it.

    Raises:
        SystemExit: If the mp4 fails to encode.
    """
    # Persist frames + state/action traces before encoding so an ffmpeg hiccup never costs the
    # ~22-min CPU render again — a re-encode, side-by-side, or re-analysis can reload these.
    npz = out.with_suffix(".npz")
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, frames=np.stack(frames), states=states, actions=actions)
    _log(f"saved {len(frames)} frames + traces to {npz}")
    analyze_jitter(actions, states, chunk=50)
    if write_mp4(frames, out, DEFAULT_FPS):
        _log(f"wrote {out} ({len(frames)} frames @ {DEFAULT_FPS} fps)")
    else:
        _log("failed to write mp4")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="hugginggeorge/smolvla-single-pawn-overfit")
    parser.add_argument("--revision", default="008000")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source", default=None, help="held-out start square, e.g. g7")
    parser.add_argument("--target", default=None, help="target square, e.g. g6")
    parser.add_argument(
        "--moves",
        default=None,
        help="batch mode: a comma-separated list of source:target pairs (e.g. "
        "'c7:c6,d6:d5,g7:g6'). Loads the policy once, drives each move, and reports the success "
        "rate. One mp4+npz per move is written next to --out, suffixed by the move.",
    )
    parser.add_argument("--out", type=Path, default=Path("runpod/artifacts/rollout-ep0.mp4"))
    args = parser.parse_args()

    if args.moves is not None:
        moves = parse_moves(args.moves)
        _log(f"batch of {len(moves)} move(s); loading policy once...")
        policy, pre, post = _load_policy(args.repo_id, args.revision, args.device)
        checks: list[tuple[tuple[str, str], PlacementCheck]] = []
        for source, target in moves:
            spec = _spec_for(args.episode, source, target)
            frames, states, actions, check = drive(policy, pre, post, spec, args.max_frames)
            _finish_move(frames, states, actions, _artifact_path(args.out, source, target))
            checks.append(((source, target), check))
        passed = sum(1 for _, c in checks if c.placed)
        _log(f"=== success rate: {passed}/{len(checks)} ===")
        for (source, target), check in checks:
            _log(
                f"  {source}->{target}: {'SUCCESS' if check.placed else 'FAIL'} "
                f"(ended {check.final_square}, settled={check.settled})"
            )
        if passed < len(checks):
            sys.exit(1)
        return

    frames, states, actions, check = rollout(
        args.repo_id,
        args.revision,
        args.episode,
        args.max_frames,
        args.device,
        source=args.source,
        target=args.target,
    )
    _finish_move(frames, states, actions, args.out)
    if not check.placed:
        _log(f"rollout did not place the piece on target (ended {check.final_square})")
        sys.exit(1)


if __name__ == "__main__":
    main()
