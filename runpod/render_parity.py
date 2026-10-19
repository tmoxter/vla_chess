"""EGL↔osmesa render parity probe.

Renders the dataset's wrist camera for one fixed, deterministic scene and saves the raw uint8
frame. Run it once locally under `MUJOCO_GL=osmesa` and once on a GPU pod under `MUJOCO_GL=egl`,
then `compare` the two frames: the backends will never be bit-identical (software vs GPU
rasterisation differ at edges), but the datagen path is only trustworthy if they agree to within a
few grey levels over the vast majority of pixels. No physics is stepped — `make_data` rests and
forward-evaluates the arm, so the frame is a pure function of (model, camera, pose) and the only
thing that varies between the two runs is the GL backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# A fixed board pose + single black pawn on d7, rendered from the wrist camera the dataset carries.
PARITY_SQUARE = "d7"


def render_frame() -> np.ndarray:
    """Render the wrist camera for the fixed parity scene.

    Returns:
        The (H, W, 3) uint8 frame, copied out of the renderer's reused buffer.
    """
    from vla_chess.dataset.episode import WRIST_CAMERA
    from vla_chess.sensor.render import Renderer
    from vla_chess.sim.positions import single_pawn_position
    from vla_chess.sim.scene import (
        DEFAULT_BOARD_POSE,
        SceneDescription,
        build_scene,
        make_data,
    )

    # arm_qpos / pinch_offset keep their defaults (RESTING_ARM_QPOS) — the parity frame only needs
    # a fixed, forward-evaluated pose, identical on both backends.
    description = SceneDescription(
        board_pose=DEFAULT_BOARD_POSE,
        pieces=single_pawn_position(PARITY_SQUARE),
    )
    model, index = build_scene(description)
    data = make_data(model, index)
    with Renderer(model, (WRIST_CAMERA,)) as renderer:
        frame = renderer.render(data)[WRIST_CAMERA.name]
        return frame.copy()


def compare(a: np.ndarray, b: np.ndarray, tol: int) -> dict[str, float]:
    """Quantify how far two uint8 frames diverge.

    Args:
        a: First frame, (H, W, 3) uint8.
        b: Second frame, same shape.
        tol: Per-channel grey-level difference below which a pixel counts as matching.

    Returns:
        Max/mean absolute difference and the fraction of channel values off by more than `tol`.
    """
    if a.shape != b.shape:
        raise ValueError(f"frame shapes differ: {a.shape} vs {b.shape}")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "frac_over_tol": float((diff > tol).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    """Render a parity frame or compare two of them.

    Args:
        argv: Command line without the program name; `sys.argv` is read when omitted.

    Returns:
        0 on success (or a passing comparison), 1 if a comparison exceeds the thresholds.
    """
    parser = argparse.ArgumentParser(prog="render-parity")
    sub = parser.add_subparsers(dest="cmd", required=True)

    render = sub.add_parser("render", help="render the parity scene to a .npy frame")
    render.add_argument("--out", type=Path, required=True)

    cmp = sub.add_parser("compare", help="compare two parity frames")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
    cmp.add_argument("--tol", type=int, default=2, help="matching grey-level tolerance per channel")
    cmp.add_argument(
        "--max-frac", type=float, default=0.02, help="max allowed fraction of pixels over tol"
    )
    cmp.add_argument(
        "--max-mean", type=float, default=1.0, help="max allowed mean absolute difference"
    )

    args = parser.parse_args(argv)

    if args.cmd == "render":
        frame = render_frame()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, frame)
        print(f"rendered {frame.shape} {frame.dtype} -> {args.out} (sum={int(frame.sum())})")
        return 0

    stats = compare(np.load(args.a), np.load(args.b), args.tol)
    passed = stats["frac_over_tol"] <= args.max_frac and stats["mean_abs"] <= args.max_mean
    print(
        f"max_abs={stats['max_abs']:.0f} mean_abs={stats['mean_abs']:.4f} "
        f"frac_over_tol(>{args.tol})={stats['frac_over_tol']:.5f} "
        f"-> {'PASS' if passed else 'FAIL'} "
        f"(limits: mean<={args.max_mean}, frac<={args.max_frac})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
