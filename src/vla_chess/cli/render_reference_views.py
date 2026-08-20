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
    """Stand the arm at one configuration and settle the kinematics on it."""
    for name, value in zip(ARM_JOINT_NAMES, qpos, strict=True):
        data.qpos[model.jnt_qposadr[index.arm_joint_id[name]]] = value
    if gripper is not None:
        data.qpos[model.jnt_qposadr[index.gripper_joint_id]] = gripper
    mujoco.mj_forward(model, data)


def render_observation_views(board_poses: Sequence[BoardPose]) -> npt.NDArray[np.uint8]:
    """Render what the wrist camera sees at each of several board distances.

    Args:
        board_poses: The placements to compare, one row of the image each.

    Returns:
        The rows stacked into one image: per pose, a caption naming its board distance above
        the resting view and the birds-eye view side by side.
    """
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
    """Render the committed reference grasp of each square from three fixed angles.

    The jaws are drawn wide open — the pose they have to be in on the way down, and the one
    that shows whether the fingers straddle the piece rather than the square beside it.

    Args:
        board_pose: Placement the grasps are rendered against, which has to be the one they
            were solved at for the poses to land on their squares.
        squares: Algebraic names of the squares to show, one row of the image each.

    Returns:
        The rows stacked into one image: per square, a caption above the side, front and top
        views side by side.

    Raises:
        SystemExit: If any named square has no reference grasp to render.
    """
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
    """Parse the reference-view renderer's command line.

    Args:
        argv: Command line to parse, without the program name. `sys.argv` is read when
            omitted.

    Returns:
        The parsed options: where to write the images, which board distances to stack for
        comparison, and which squares' grasps to show.
    """
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
    """Render both reference images and write them into the output directory.

    Args:
        argv: Command line to parse, without the program name. `sys.argv` is read when
            omitted.

    Returns:
        Process exit status, always 0: these images are regenerated to be looked at, not
        checked against a threshold.
    """
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
