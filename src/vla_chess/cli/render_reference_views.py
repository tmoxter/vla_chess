import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.grasp.solver import load_reference_grasps
from vla_chess.sensor.render import CameraSpec, FreeCameraPose, Renderer
from vla_chess.sim.positions import starting_position
from vla_chess.sim.scene import (
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
from vla_chess.viz.imaging import caption, write_png

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
