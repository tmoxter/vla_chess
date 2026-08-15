import argparse
import struct
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.cli.solve_reference_grasps import starting_position
from vla_chess.grasp import load_reference_grasps
from vla_chess.render import CameraSpec, FreeCameraPose, Renderer
from vla_chess.scene import (
    ARM_JOINT_NAMES,
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    RESTING_ARM_QPOS,
    WRIST_CAMERA_NAME,
    BoardPose,
    SceneDescription,
    SceneIndex,
    build_scene,
    make_data,
)

DEFAULT_OUTPUT_DIR: Final = Path("docs/reference-views")
BACK_RANK_SQUARES: Final = ("c8", "d8", "e8")
PANEL: Final = 460

# Fixed free cameras, not the wrist camera: a grasp pose has to be judged from outside the arm,
# and these three angles are what makes "are the fingers really down, on really that square"
# answerable by eye. Positions are in world metres and were chosen to frame the near half of
# the board with the arm's whole doubled-back configuration visible.
GRASP_CAMERAS: Final = (
    ("side", (0.10, -0.55, 0.30), (0.16, 0.0, 0.18)),
    ("front", (0.66, 0.0, 0.30), (0.16, 0.0, 0.16)),
    ("top", (0.18, -0.02, 0.80), (0.18, 0.0, 0.10)),
)

# 5x7 bitmap glyphs, drawn at 2x, covering only the characters the captions use. A caption
# burned into the image is what keeps a reference render self-identifying once it is out of
# this repo and in a chat window.
_GLYPHS: Final = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "b": ("10000", "10000", "10110", "11001", "10001", "10001", "11110"),
    "c": ("00000", "00000", "01110", "10001", "10000", "10001", "01110"),
    "d": ("00001", "00001", "01101", "10011", "10001", "10001", "01111"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01110"),
    "g": ("00000", "00000", "01111", "10001", "01111", "00001", "01110"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    "s": ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
    "t": ("01000", "01000", "11110", "01000", "01000", "01001", "00110"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
    "y": ("00000", "00000", "10001", "10001", "01111", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    " ": ("00000",) * 7,
}


def write_png(path: Path, image: npt.NDArray[np.uint8]) -> None:
    """Write an (h, w, 3) uint8 array as an RGB PNG, using only the standard library."""
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def caption(width: int, text: str, height: int = 22) -> npt.NDArray[np.uint8]:
    """A black strip carrying `text` in white, to sit above a row of panels."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    x = 4
    for character in text:
        glyph = _GLYPHS.get(character, _GLYPHS[" "])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    bar[6 + row * 2 : 8 + row * 2, x + column * 2 : x + column * 2 + 2] = 255
        x += 12
    return bar


def _apply(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    qpos: Sequence[float],
    gripper: float | None = None,
) -> None:
    for name, value in zip(ARM_JOINT_NAMES, qpos, strict=True):
        data.qpos[model.jnt_qposadr[index.arm_joint_id[name]]] = value
    if gripper is not None:
        data.qpos[model.jnt_qposadr[index.gripper_joint_id]] = gripper
    mujoco.mj_forward(model, data)


def render_observation_views(board_poses: Sequence[BoardPose]) -> npt.NDArray[np.uint8]:
    """One captioned row per board pose: the wrist camera at rest, then at the birds-eye pose."""
    rows = []
    for board_pose in board_poses:
        model, index = build_scene(
            SceneDescription(board_pose=board_pose, pieces=starting_position())
        )
        data = make_data(model, index)
        frames = []
        with Renderer(model, [CameraSpec(WRIST_CAMERA_NAME, 512, 512)]) as renderer:
            for qpos in (RESTING_ARM_QPOS, BIRDS_EYE_ARM_QPOS):
                _apply(model, data, index, qpos)
                # Copied, not referenced: the renderer hands back a view of a buffer it reuses
                # on the next call, so keeping the reference silently yields two copies of the
                # last pose rendered.
                frames.append(np.array(renderer.render(data)[WRIST_CAMERA_NAME], copy=True))
        row = np.hstack(frames)
        rows.append(np.vstack([caption(row.shape[1], f"x={board_pose.x:.3f}"), row]))
    return np.ascontiguousarray(np.vstack(rows))


def render_grasp_views(board_pose: BoardPose, squares: Sequence[str]) -> npt.NDArray[np.uint8]:
    """One captioned row per square: the reference grasp seen from the side, front and above."""
    model, index = build_scene(SceneDescription(board_pose=board_pose, pieces=starting_position()))
    data = make_data(model, index)
    grasps = load_reference_grasps()
    cameras = [
        CameraSpec(name, PANEL, PANEL, fovy=45, pose=FreeCameraPose(eye, look_at))
        for name, eye, look_at in GRASP_CAMERAS
    ]

    rows = []
    with Renderer(model, cameras) as renderer:
        for square in squares:
            if square not in grasps:
                raise SystemExit(f"no reference grasp for {square}")
            # Jaws wide open: the pose the gripper has to be in on the way down, and the one
            # that shows whether the fingers straddle the piece rather than the square beside it.
            _apply(model, data, index, grasps[square].qpos, index.gripper_joint_range[1])
            frames = renderer.render(data)
            row = np.hstack([np.asarray(frames[camera.name]) for camera in cameras])
            rows.append(np.vstack([caption(row.shape[1], f"{square} x={board_pose.x:.3f}"), row]))
    return np.ascontiguousarray(np.vstack(rows))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the reference-view renderer's command line."""
    parser = argparse.ArgumentParser(
        prog="render-reference-views",
        description="Re-render the committed visual reference views of the scene's board pose "
        "and back-rank reference grasps.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--compare-board-x",
        type=float,
        nargs="+",
        default=[0.280, DEFAULT_BOARD_POSE.x],
        help="board distances to stack in the observation-view image, for comparing a "
        "candidate placement against the current one",
    )
    parser.add_argument("--squares", nargs="+", default=list(BACK_RANK_SQUARES))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render both reference images and write them into the output directory."""
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    poses = [
        BoardPose(x, DEFAULT_BOARD_POSE.y, DEFAULT_BOARD_POSE.yaw, DEFAULT_BOARD_POSE.square_pitch)
        for x in args.compare_board_x
    ]
    observation = args.out_dir / "observation-views.png"
    write_png(observation, render_observation_views(poses))
    print(f"wrote {observation}", flush=True)

    grasps = args.out_dir / "back-rank-grasps.png"
    write_png(grasps, render_grasp_views(DEFAULT_BOARD_POSE, args.squares))
    print(f"wrote {grasps}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
