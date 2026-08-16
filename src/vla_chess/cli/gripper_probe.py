import argparse
from collections.abc import Sequence

import mujoco
import numpy as np

from vla_chess.calibration import measure_jaw_faces
from vla_chess.collision import piece_clearance
from vla_chess.grasp import GRASP_HEIGHT, VerticalGraspSolver
from vla_chess.scene import (
    ARM_JOINT_NAMES,
    FIXED_FACE_INSET,
    GRIPPER_BODY_NAME,
    JAW_GAP_TABLE,
    MOVING_JAW_BODY_NAME,
    PieceColor,
    PiecePlacement,
    PieceType,
    SceneDescription,
    build_scene,
    denormalize_gripper,
    grip_command,
    grip_radius,
    jaw_gap,
    make_data,
    pinch_offset_for,
)

DEFAULT_SQUARE = "d7"
PROBE_PIECE = PieceType.PAWN


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the gripper probe's command line."""
    parser = argparse.ArgumentParser(
        prog="gripper-probe",
        description="Measure where the jaws' faces stand, both as the engine collides them and"
        " as the meshes draw them, and report what scene.py pins on that.",
    )
    parser.add_argument("--square", default=DEFAULT_SQUARE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fine",
        action="store_true",
        help="sweep the jaw at 0.025 rad instead of at the pinned table's own angles",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Report the jaw geometry the hand's grip sizing is derived from."""
    args = parse_args(argv)
    radius = grip_radius(PROBE_PIECE, GRASP_HEIGHT)
    desc = SceneDescription(
        pieces=(PiecePlacement(args.square, PROBE_PIECE, PieceColor.BLACK),),
        pinch_offset=pinch_offset_for(radius),
    )
    model, index = build_scene(desc)
    data = make_data(model, index)
    solver = VerticalGraspSolver(model, data, index)
    pose = solver.solve_square(
        args.square,
        index.board,
        index.board_pose,
        np.random.default_rng(args.seed),
        height=GRASP_HEIGHT,
    )
    if not pose.is_within_tolerance():
        print(f"no fingers-down pose over {args.square}: {pose.tcp_error * 1000:.1f} mm off")
        return 1
    arm = np.array(pose.qpos, dtype=np.float64)
    print(
        f"probing over {args.square} with a {PROBE_PIECE.value} ({radius * 1000:.2f} mm), "
        f"tcp error {pose.tcp_error * 1000:.3f} mm\n"
    )

    angles = (
        np.round(np.arange(-0.15, 0.351, 0.025), 4)
        if args.fine
        else np.array([angle for angle, _ in JAW_GAP_TABLE])
    )
    print("faces along the closing axis, measured from `gripperframe` (mm):")
    print(
        f"{'jaw qpos':>9}  {'fixed':>7}  {'fixed drawn':>11}  "
        f"{'gap':>7}  {'gap drawn':>9}  {'gap pinned':>10}"
    )
    measured: list[tuple[float, float]] = []
    insets: list[float] = []
    for angle in angles:
        hulls = measure_jaw_faces(model, data, index, arm, float(angle), args.square, radius)
        drawn = measure_jaw_faces(
            model, data, index, arm, float(angle), args.square, radius, drawn=True
        )
        measured.append((float(angle), hulls.gap))
        insets.append(hulls.fixed)
        print(
            f"{angle:9.3f}  {hulls.fixed * 1000:7.2f}  {drawn.fixed * 1000:11.2f}  "
            f"{hulls.gap * 1000:7.2f}  {drawn.gap * 1000:9.2f}  "
            f"{jaw_gap(float(angle)) * 1000:10.2f}"
        )

    drift = max(abs(jaw_gap(angle) - gap) for angle, gap in measured)
    print(f"\nFIXED_FACE_INSET: measured {np.mean(insets):.5f}, pinned {FIXED_FACE_INSET:.5f}")
    print(f"JAW_GAP_TABLE: worst disagreement with the pinned table {drift * 1000:.2f} mm")
    print("as a table:")
    for angle, gap in measured:
        print(f"    ({angle:.3f}, {gap:.5f}),")

    grip = grip_command(radius, index.gripper_joint_range)
    data.qpos[[model.jnt_qposadr[index.arm_joint_id[n]] for n in ARM_JOINT_NAMES]] = arm
    data.qpos[model.jnt_qposadr[index.gripper_joint_id]] = denormalize_gripper(
        grip, index.gripper_joint_range
    )
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    piece = index.piece_body_id[args.square]
    print(f"\nat the sized grip command ({grip:.2f}/100), drawn clearance to the piece:")
    for name in (GRIPPER_BODY_NAME, MOVING_JAW_BODY_NAME):
        body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        clearance = piece_clearance(model, data, piece, frozenset({body}))
        print(f"  {name:24s} {clearance * 1000:+7.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
