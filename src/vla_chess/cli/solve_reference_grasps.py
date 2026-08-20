import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from vla_chess.grasp.solver import (
    GRASP_HEIGHT,
    MAX_APPROACH_DEG,
    MAX_TCP_ERROR,
    REFERENCE_BOARD_POSE,
    REFERENCE_GRASPS_PATH,
    VerticalGraspSolver,
)
from vla_chess.sim.collision import baseline_self_pairs, contacts_at_pose
from vla_chess.sim.positions import starting_position
from vla_chess.sim.scene import (
    ARM_JOINT_NAMES,
    FILES,
    RANKS,
    BoardPose,
    SceneDescription,
    build_scene,
    make_data,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the reference-grasp solver's command line.

    Args:
        argv: Command line to parse, without the program name. `sys.argv` is read when
            omitted.

    Returns:
        The parsed options: the board distance to solve at, the seed, where the asset goes,
        and whether to print the result instead of writing it.
    """
    parser = argparse.ArgumentParser(
        prog="solve-reference-grasps",
        description="Solve one fingers-down grasp pose per board square and write them to the "
        "reference asset.",
    )
    parser.add_argument(
        "--board-x",
        type=float,
        default=REFERENCE_BOARD_POSE.x,
        help="board centre distance to solve at; for exploring alternatives, since the "
        "committed asset is the scene's own board pose",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REFERENCE_GRASPS_PATH)
    parser.add_argument(
        "--no-write", action="store_true", help="print the summary instead of writing the asset"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Solve a vertical grasp for every square and record the ones that meet tolerance.

    Each square is solved twice, unscreened and then screened for collisions, so the two ways
    a square can fail stay apart: one the arm cannot reach fingers-down at all, and one it can
    only reach by folding through itself. Reporting both as "unreachable" would hide that.

    Args:
        argv: Command line to parse, without the program name. `sys.argv` is read when
            omitted.

    Returns:
        Process exit status, always 0: squares with no clean grasp are reported and left out
        of the asset rather than failing the run.
    """
    args = parse_args(argv)
    board_pose = BoardPose(
        args.board_x,
        REFERENCE_BOARD_POSE.y,
        REFERENCE_BOARD_POSE.yaw,
        REFERENCE_BOARD_POSE.square_pitch,
    )
    model, index = build_scene(SceneDescription(board_pose=board_pose, pieces=starting_position()))
    data = make_data(model, index)
    solver = VerticalGraspSolver(model, data, index)
    rng = np.random.default_rng(args.seed)
    baseline = baseline_self_pairs(model, data, index, np.random.default_rng(args.seed))

    def is_clear(square: str) -> Callable[[npt.NDArray[np.float64]], bool]:
        """Reject poses that reach the square by folding the arm through itself or the pieces.

        Screening here rather than after the fact is what makes the asset's promise true: a
        recorded pose is one the arm can actually hold, not merely one whose TCP lands in the
        right place. The jaws are checked wide open, the pose they have to be in on the way
        down, and the piece being picked up is exempt from the hand's own contacts.

        Args:
            square: Algebraic name of the square being solved, whose piece the hand is allowed
                to touch.

        Returns:
            A screen the solver applies to each converged pose.
        """

        def accept(qpos: npt.NDArray[np.float64]) -> bool:
            """Report whether the arm holds this pose touching nothing but the piece."""
            return contacts_at_pose(
                model,
                data,
                index,
                qpos,
                grasped_square=square,
                baseline=baseline,
                gripper_qpos=index.gripper_joint_range[1],
            ).is_clear()

        return accept

    grasps: dict[str, dict[str, object]] = {}
    out_of_reach: list[str] = []
    blocked: list[str] = []
    for rank in reversed(RANKS):
        row = []
        for file in FILES:
            square = f"{file}{rank}"
            # Unscreened first, to tell the two failures apart. A square the arm cannot reach
            # fingers-down and one it can only reach by folding through itself are different
            # problems with different fixes, and reporting both as "unreachable" hides that.
            reach = solver.solve_square(square, index.board, board_pose, rng)
            if not reach.is_within_tolerance():
                out_of_reach.append(square)
                row.append(f"{square}:-{reach.tcp_error * 1000:.0f}mm")
                continue
            pose = solver.solve_square(
                square, index.board, board_pose, rng, accept=is_clear(square)
            )
            if not pose.is_within_tolerance():
                blocked.append(square)
                row.append(f"{square}:blocked")
                continue
            grasps[square] = {
                "qpos": [float(v) for v in pose.qpos],
                "tcp_error": float(pose.tcp_error),
                "approach_deg": float(pose.approach_deg),
            }
            row.append(f"{square}:ok")
        print(f"  rank {rank}: " + "  ".join(row), flush=True)

    print(f"{len(grasps)}/64 squares have a clean fingers-down grasp", flush=True)
    if out_of_reach:
        print(f"out of reach fingers-down: {', '.join(out_of_reach)}", flush=True)
    if blocked:
        print(
            f"reachable but every solution collides: {', '.join(blocked)}",
            flush=True,
        )

    payload = {
        "metadata": {
            "board_pose": {
                "x": board_pose.x,
                "y": board_pose.y,
                "yaw": board_pose.yaw,
                "square_pitch": board_pose.square_pitch,
            },
            "grasp_height": GRASP_HEIGHT,
            "joint_order": list(ARM_JOINT_NAMES),
            "max_tcp_error": MAX_TCP_ERROR,
            "max_approach_deg": MAX_APPROACH_DEG,
            "seed": args.seed,
        },
        "grasps": grasps,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.no_write:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
