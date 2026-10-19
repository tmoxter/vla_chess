"""Diagnose which constraint gates each square's grasp: reach (TCP error) or tilt (approach deg).

For every square, solve the collision-screened grasp pose at grasp height and report the *best*
attempt's TCP error and approach angle, ignoring the acceptance thresholds. This separates "the
arm cannot get its tool point to the square at all" (large TCP error — tilt won't help) from "the
arm reaches the square but only with a tilted wrist" (small TCP error, large approach angle — a
relaxed approach tolerance opens it up).
"""

from __future__ import annotations

import numpy as np

from vla_chess.grasp.solver import GRASP_HEIGHT, VerticalGraspSolver
from vla_chess.sim.collision import (
    baseline_self_pairs,
    contacts_at_pose,
    hand_body_ids,
    piece_clearance,
)
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    SceneDescription,
    build_scene,
    make_data,
)

FILES = "abcdefgh"


def probe_square(square: str) -> tuple[float, float]:
    """Best-attempt TCP error (mm) and approach angle (deg) for a collision-screened grasp."""
    desc = SceneDescription(
        board_pose=DEFAULT_BOARD_POSE,
        pieces=single_pawn_position(square),
        arm_qpos=BIRDS_EYE_ARM_QPOS,
    )
    model, index = build_scene(desc)
    data = make_data(model, index)
    solver = VerticalGraspSolver(model, data, index)
    baseline = baseline_self_pairs(model, data, index, np.random.default_rng(0))
    hands = hand_body_ids(model)
    piece_body = index.piece_body_id.get(square)
    opening = index.gripper_joint_range[1]

    def accept(qpos: np.ndarray) -> bool:
        contacts = contacts_at_pose(
            model, data, index, qpos, grasped_square=square, baseline=baseline, gripper_qpos=opening
        )
        if not contacts.is_clear():
            return False
        if piece_body is None:
            return True
        return piece_clearance(model, data, piece_body, hands) > 0.0

    pose = solver.solve_square(
        square,
        index.board,
        index.board_pose,
        np.random.default_rng(0),
        height=GRASP_HEIGHT,
        accept=accept,
    )
    return pose.tcp_error * 1000.0, pose.approach_deg


def main() -> int:
    print("Best-attempt grasp per square (single-pawn board). '.'=unreached(>10mm), else tilt deg:")
    print("     " + "     ".join(FILES))
    for rank in range(8, 0, -1):
        cells = []
        for f in FILES:
            sq = f"{f}{rank}"
            try:
                err_mm, deg = probe_square(sq)
            except Exception:
                cells.append(" ERR ")
                continue
            if err_mm > 10.0:
                cells.append("  .  ")
            else:
                cells.append(f"{deg:4.0f}°")
        print(f" {rank}  " + " ".join(cells))
    print("\n(cells show approach angle off vertical for squares the TCP reaches within 10mm;")
    print(" '.' = TCP cannot reach the square at all, so tilt tolerance won't help it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
