import argparse
import sys
from collections.abc import Sequence

import mujoco
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from vla_chess.analysis.survey import LIMIT_MARGIN, SurveyFraming, framing, level_rolls
from vla_chess.grasp.solver import joint_limits
from vla_chess.sim.positions import starting_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    BoardPose,
    SceneDescription,
    SceneIndex,
    build_scene,
    make_data,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for a birds-eye survey pose that frames the whole board.",
    )
    parser.add_argument("--board-x", type=float, default=DEFAULT_BOARD_POSE.x)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-pitch",
        type=float,
        default=75.0,
        help="shallowest camera pitch below horizontal to accept, in degrees",
    )
    parser.add_argument("--max-pitch", type=float, default=90.0)
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.06,
        help="worst-corner margin floor, as a fraction of the half-frame",
    )
    return parser.parse_args(argv)


def _best_framing(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    board_pose: BoardPose,
    free: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    rays: bool = False,
) -> tuple[npt.NDArray[np.float64], SurveyFraming] | None:
    """The better of the two level-horizon rolls for one set of the other four joints.

    Both roots have to be tried: `wrist_roll` is not the camera's optical axis, so one of
    them swings the camera off the board entirely rather than merely rolling the image.
    """
    joints = np.clip(free, low[:4], high[:4])
    best: tuple[npt.NDArray[np.float64], SurveyFraming] | None = None
    for roll in level_rolls(model, data, index, np.concatenate([joints, [0.0]])):
        if not low[4] <= roll <= high[4]:
            continue
        candidate = np.concatenate([joints, [roll]])
        view = framing(model, data, index, index.board, board_pose, candidate, rays=rays)
        if best is None or view.worst_margin > best[1].worst_margin:
            best = (candidate, view)
    return best


def _report(
    label: str, qpos: npt.NDArray[np.float64], empty: SurveyFraming, full: SurveyFraming
) -> None:
    """Print one pose, how it frames the board, and what it can see of a full board."""
    print(f"{label}: qpos {np.round(qpos, 9).tolist()}")
    print(
        f"  worst margin {empty.worst_margin * 100:+.2f}% at {empty.worst_square}, "
        f"pitch {empty.pitch_deg:.1f} deg, height {empty.height * 100:.1f} cm, "
        f"horizon tilt {empty.horizon_tilt_deg:+.4f} deg"
    )
    print(
        f"  {64 - len(empty.out_of_frame)}/64 in frame, {empty.unoccluded}/64 unoccluded by the "
        f"arm, {full.unoccluded}/64 unoccluded from the starting position"
    )
    if empty.occluded:
        print(f"  hidden by the arm: {', '.join(empty.occluded)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Search arm configurations for the survey view that sees the most of the whole board.

    Framing the board is the constraint, not the objective: several poses put all 64 squares
    inside the image, and they differ by up to 17 squares in how many the arm's own body then
    hides. Maximising margin alone picks a wide, oblique view that sees less than the pose it
    replaces.
    """
    args = parse_args(argv)
    board_pose = BoardPose(
        args.board_x,
        DEFAULT_BOARD_POSE.y,
        DEFAULT_BOARD_POSE.yaw,
        DEFAULT_BOARD_POSE.square_pitch,
    )
    # Occlusion is searched on an empty board, so the objective answers "what does the arm's
    # own body hide" rather than "what do the pieces hide", which changes with every position.
    model, index = build_scene(SceneDescription(board_pose=board_pose, pieces=()))
    data = make_data(model, index)
    full_model, full_index = build_scene(
        SceneDescription(board_pose=board_pose, pieces=starting_position())
    )
    full_data = make_data(full_model, full_index)
    low, high = joint_limits(model, index)
    low, high = low + LIMIT_MARGIN, high - LIMIT_MARGIN

    def evaluate(
        free: npt.NDArray[np.float64], rays: bool = False
    ) -> tuple[npt.NDArray[np.float64], SurveyFraming] | None:
        return _best_framing(model, data, index, board_pose, free, low, high, rays=rays)

    def cost(free: npt.NDArray[np.float64]) -> float:
        found = evaluate(free)
        if found is None:
            return 50.0
        candidate, view = found
        penalty = 2.0 * float(
            max(0.0, args.min_pitch - view.pitch_deg) + max(0.0, view.pitch_deg - args.max_pitch)
        )
        if not np.isfinite(view.worst_margin):
            return 20.0 + len(view.out_of_frame) + penalty
        if view.out_of_frame:
            return 5.0 - view.worst_margin + penalty
        seen = framing(model, data, index, index.board, board_pose, candidate, rays=True).unoccluded
        return -seen / 64.0 - min(view.worst_margin, float(args.min_margin)) + penalty

    def describe(qpos: npt.NDArray[np.float64]) -> tuple[SurveyFraming, SurveyFraming]:
        empty = framing(model, data, index, index.board, board_pose, qpos, rays=True)
        full = framing(
            full_model, full_data, full_index, full_index.board, board_pose, qpos, rays=True
        )
        return empty, full

    current = np.array(BIRDS_EYE_ARM_QPOS, dtype=float)
    _report("current", current, *describe(current))

    rng = np.random.default_rng(args.seed)
    origin = np.array(BIRDS_EYE_ARM_QPOS[:4], dtype=float)
    starts = [origin] + [origin + rng.normal(0.0, 0.2, 4) for _ in range(args.restarts - 1)]
    best: tuple[float, npt.NDArray[np.float64], SurveyFraming] | None = None
    for start in starts:
        result = minimize(
            cost,
            np.clip(start, low[:4], high[:4]),
            method="Nelder-Mead",
            options={"maxiter": 200, "maxfev": 400, "fatol": 1e-9},
        )
        found = evaluate(result.x, rays=True)
        if found is None:
            continue
        candidate, view = found
        if not view.frames_whole_board():
            continue
        if not args.min_pitch - 0.3 <= view.pitch_deg <= args.max_pitch + 0.3:
            continue
        if best is None or view.unoccluded > best[0]:
            best = (float(view.unoccluded), candidate.copy(), view)

    if best is None:
        print("no pose framed the whole board within the pitch band", file=sys.stderr)
        return 1
    _report("\nbest", best[1], *describe(best[1]))
    if best[2].worst_margin < args.min_margin:
        print(
            f"best margin {best[2].worst_margin * 100:.2f}% is under the "
            f"{args.min_margin * 100:.0f}% floor",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
