import json

import mujoco
import numpy as np
import numpy.typing as npt
import pytest

from vla_chess.control.planner import ArmQpos, _above
from vla_chess.grasp.solver import (
    GRASP_HEIGHT,
    MAX_APPROACH_DEG,
    MAX_TCP_ERROR,
    MAX_TILT_DEG,
    MIN_LIMIT_MARGIN,
    REFERENCE_BOARD_POSE,
    REFERENCE_GRASPS_PATH,
    GraspPose,
    VerticalGraspSolver,
    joint_limits,
    load_reference_grasps,
    reference_metadata,
)
from vla_chess.sim.scene import (
    ARM_JOINT_NAMES,
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    PieceColor,
    PiecePlacement,
    PieceType,
    SceneDescription,
    SceneIndex,
    build_scene,
    grip_radius,
    make_data,
    pinch_offset_for,
)

# A square with room around it, so the poses these tests compare differ in roll and in nothing
# else.
PREFERENCE_SQUARE = "e7"

# A configuration to measure travel from, chosen so the arithmetic in a cost is readable.
ORIGIN = np.zeros(5)

# The nearest squares the arm cannot reach fingers-down, so the frontier is pinned from the
# far side as well as the near one; see docs/grasp-poses.md.
OUT_OF_REACH_FRONTIER = {"a5", "h5", "d4", "e4"}


@pytest.fixture(scope="module")
def grasps() -> dict[str, GraspPose]:
    return load_reference_grasps()


class TestGraspPose:
    def test_pose_inside_both_thresholds_is_within_tolerance(self) -> None:
        pose = GraspPose("e7", (0.0, 0.0, 0.0, 0.0, 0.0), MAX_TCP_ERROR, MAX_APPROACH_DEG)
        assert pose.is_within_tolerance()

    def test_pose_off_target_is_not_within_tolerance(self) -> None:
        pose = GraspPose("e7", (0.0, 0.0, 0.0, 0.0, 0.0), MAX_TCP_ERROR * 2, 0.0)
        assert not pose.is_within_tolerance()

    def test_tilted_pose_is_not_within_tolerance(self) -> None:
        pose = GraspPose("e7", (0.0, 0.0, 0.0, 0.0, 0.0), 0.0, MAX_APPROACH_DEG * 2)
        assert not pose.is_within_tolerance()


class TestReferenceAsset:
    def test_asset_exists(self) -> None:
        assert REFERENCE_GRASPS_PATH.is_file()

    def test_joint_order_matches_the_scene(self) -> None:
        assert reference_metadata()["joint_order"] == list(ARM_JOINT_NAMES)

    def test_board_pose_matches_the_module_constant(self) -> None:
        pose = reference_metadata()["board_pose"]
        assert isinstance(pose, dict)
        assert pose["x"] == pytest.approx(REFERENCE_BOARD_POSE.x)
        assert pose["y"] == pytest.approx(REFERENCE_BOARD_POSE.y)
        assert pose["yaw"] == pytest.approx(REFERENCE_BOARD_POSE.yaw)
        assert pose["square_pitch"] == pytest.approx(REFERENCE_BOARD_POSE.square_pitch)

    def test_grasp_height_matches_the_module_constant(self) -> None:
        assert reference_metadata()["grasp_height"] == pytest.approx(GRASP_HEIGHT)

    def test_reference_pose_is_the_scene_default_board_pose(self) -> None:
        # The inverse of a test that used to live here, which asserted the two *differed*. They
        # are one pose now: a reference set solved at its own board distance does not apply to
        # the scene anyone actually builds.
        assert REFERENCE_BOARD_POSE == DEFAULT_BOARD_POSE

    def test_every_pose_has_one_angle_per_arm_joint(self, grasps: dict[str, GraspPose]) -> None:
        assert all(len(pose.qpos) == len(ARM_JOINT_NAMES) for pose in grasps.values())

    def test_every_pose_meets_tolerance(self, grasps: dict[str, GraspPose]) -> None:
        assert all(pose.is_within_tolerance() for pose in grasps.values())

    def test_keys_agree_with_the_square_each_pose_names(self, grasps: dict[str, GraspPose]) -> None:
        assert all(square == pose.square for square, pose in grasps.items())

    def test_black_home_and_opening_ranks_are_covered(self, grasps: dict[str, GraspPose]) -> None:
        # Ranks 6-8 are what playing black's opening needs: pick up from 7-8, put down on 6.
        # All 24 of them, with no exception carved out — d8 and e8 are in here on purpose (see
        # the next test), and an exception list creeping back is the regression to catch.
        needed = {f"{file}{rank}" for file in "abcdefgh" for rank in "678"}
        assert needed <= set(grasps)

    def test_the_black_queen_and_king_can_be_lifted_off_their_home_squares(
        self, grasps: dict[str, GraspPose]
    ) -> None:
        # d8 and e8 are the squares the board distance is set by. They sit directly in front of
        # the shoulder, so the arm doubles back over itself to point the fingers down at them,
        # and the poses that exist for them only clear the shoulder because the board was moved
        # out to 0.300; nearer than ~0.298 every solution folds into it. Pinned because it is
        # load-bearing for move planning and because it would go quietly: nothing else fails
        # when the queen and king become unliftable.
        assert {"d8", "e8"} <= set(grasps)

    def test_the_out_of_reach_frontier_is_absent(self, grasps: dict[str, GraspPose]) -> None:
        # The cost side of that board distance, pinned so it stays a decision. a5/h5 miss by
        # 21 mm and d4/e4 by 19 mm fingers-down; d4 and e4 were reachable at 0.28, before the
        # board moved out. If these come back, the board moved.
        assert OUT_OF_REACH_FRONTIER.isdisjoint(grasps)

    def test_unreachable_squares_are_absent_rather_than_recorded(self) -> None:
        payload = json.loads(REFERENCE_GRASPS_PATH.read_text())
        assert "a1" not in payload["grasps"]


Scene = tuple[VerticalGraspSolver, mujoco.MjModel, SceneIndex]


@pytest.fixture(scope="module")
def solver_scene() -> Scene:
    radius = grip_radius(PieceType.PAWN, GRASP_HEIGHT)
    desc = SceneDescription(
        pieces=(PiecePlacement(PREFERENCE_SQUARE, PieceType.PAWN, PieceColor.BLACK),),
        pinch_offset=pinch_offset_for(radius),
    )
    model, index = build_scene(desc)
    data = make_data(model, index)
    return VerticalGraspSolver(model, data, index), model, index


def rolls_over(
    solver: VerticalGraspSolver, index: SceneIndex, seeds: range, prefer: ArmQpos | None
) -> list[GraspPose]:
    return [
        solver.solve_square(
            PREFERENCE_SQUARE,
            index.board,
            index.board_pose,
            np.random.default_rng(seed),
            prefer=prefer,
        )
        for seed in seeds
    ]


class TestFreeMargin:
    def test_a_pose_on_the_roll_stop_has_no_margin(self, solver_scene: Scene) -> None:
        solver, model, index = solver_scene
        lo, _ = joint_limits(model, index)
        assert solver.free_margin(lo) == pytest.approx(0.0)

    def test_a_pose_away_from_the_roll_stops_has_margin(self, solver_scene: Scene) -> None:
        solver, _, _ = solver_scene
        assert solver.free_margin(np.zeros(len(ARM_JOINT_NAMES))) > MIN_LIMIT_MARGIN

    def test_the_other_joints_do_not_count_against_a_pose(self, solver_scene: Scene) -> None:
        # A fingers-down pose over the arm's home ranks holds wrist_flex within 1.5 degrees of
        # its stop whatever the roll does, so counting every joint would rank every solution
        # clamped and leave the choice to travel alone — which is the roll stop again.
        solver, model, index = solver_scene
        _, hi = joint_limits(model, index)
        flexed = np.zeros(len(ARM_JOINT_NAMES))
        flexed[ARM_JOINT_NAMES.index("wrist_flex")] = hi[ARM_JOINT_NAMES.index("wrist_flex")]
        assert solver.free_margin(flexed) > MIN_LIMIT_MARGIN


class TestTravelCost:
    """How two solutions that reach the square equally well are ranked against each other."""

    def test_the_nearer_pose_ranks_first(self, solver_scene: Scene) -> None:
        solver, _, _ = solver_scene
        near = np.array([0.0, 0.0, 0.0, 0.0, np.radians(-11.0)])
        far = np.array([0.0, 0.0, 0.0, 0.0, np.radians(120.0)])
        assert solver.travel_cost(near, ORIGIN) < solver.travel_cost(far, ORIGIN)

    def test_the_widest_single_turn_decides_before_the_total(self, solver_scene: Scene) -> None:
        # Four joints turning a little are cheaper than one turning a lot, even when they add up
        # to more: it is the widest turn that sets how fast a fixed-duration leg has to move.
        solver, _, _ = solver_scene
        spread = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        single = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
        assert solver.travel_cost(spread, ORIGIN) < solver.travel_cost(single, ORIGIN)

    def test_a_clamped_pose_ranks_behind_a_further_free_one(self, solver_scene: Scene) -> None:
        solver, model, index = solver_scene
        lo, _ = joint_limits(model, index)
        clamped = np.array([0.0, 0.0, 0.0, 0.0, lo[ARM_JOINT_NAMES.index("wrist_roll")]])
        free = np.array([0.0, 0.0, 0.0, 0.0, np.radians(150.0)])
        assert solver.travel_cost(clamped, ORIGIN) > solver.travel_cost(free, ORIGIN)


class TestPreferredSolutions:
    """The fingers-down constraint leaves roll free, so which solution comes back is a choice."""

    def test_a_preferred_solve_still_reaches_the_square(self, solver_scene: Scene) -> None:
        solver, _, index = solver_scene
        for pose in rolls_over(solver, index, range(3), BIRDS_EYE_ARM_QPOS):
            assert pose.is_within_tolerance()

    def test_a_preference_settles_which_solution_comes_back(self, solver_scene: Scene) -> None:
        # Without one the roll is whatever the first accepted restart converged on, which is a
        # different quarter-turn from one seed to the next; with one it is the same pose every
        # time, because every seed searches the same continuum for the same nearest point.
        solver, _, index = solver_scene
        preferred = rolls_over(solver, index, range(4), BIRDS_EYE_ARM_QPOS)
        rolls = [pose.qpos[ARM_JOINT_NAMES.index("wrist_roll")] for pose in preferred]
        assert max(rolls) - min(rolls) < np.radians(1.0)

    def test_the_preferred_solution_keeps_off_the_stops(self, solver_scene: Scene) -> None:
        # The two nearest solutions to the survey pose are a degree of travel apart and one of
        # them is clamped on the roll stop, so nearness alone would pick the clamped one.
        solver, _, index = solver_scene
        for pose in rolls_over(solver, index, range(3), BIRDS_EYE_ARM_QPOS):
            assert solver.free_margin(np.asarray(pose.qpos)) >= MIN_LIMIT_MARGIN


# Squares no fingers-down pose reaches (OUT_OF_REACH_FRONTIER). All four are position-reachable
# once the wrist may tilt, but the tilt each needs is a function of the grasp geometry: with the
# pawn's pinch offset a5 grasps at a modest ~10 deg, while d4/e4 land near ~47 deg — past the
# 45 deg budget. So position-reachable is the whole frontier; graspable-within-budget is a
# subset. `a5` is the square used to assert an actual in-budget tilted grasp.
FRONTIER_SQUARES = ("a5", "d4", "e4")
IN_BUDGET_TILTED_SQUARE = "a5"


class TestTiltedGrasp:
    """The relaxed mode: solve position-first and accept the natural wrist tilt up to the cap."""

    def test_vertical_mode_leaves_the_frontier_out_of_reach(self, solver_scene: Scene) -> None:
        solver, _, index = solver_scene
        for square in FRONTIER_SQUARES:
            pose = solver.solve_square(
                square, index.board, index.board_pose, np.random.default_rng(0)
            )
            assert not pose.is_within_tolerance()

    def test_tilted_mode_reaches_the_frontier_positions(self, solver_scene: Scene) -> None:
        # Relaxing only the wrist orientation is enough to put the TCP on squares vertical could
        # not reach at all — position is met across the whole frontier.
        solver, _, index = solver_scene
        for square in FRONTIER_SQUARES:
            pose = solver.solve_square(
                square,
                index.board,
                index.board_pose,
                np.random.default_rng(0),
                weight=0.0,
                max_approach_deg=MAX_TILT_DEG,
            )
            assert pose.tcp_error <= MAX_TCP_ERROR

    def test_tilted_mode_grips_an_in_budget_square_off_vertical(self, solver_scene: Scene) -> None:
        # a5 is unreachable fingers-down but grasps within the tilt budget: a pose the vertical
        # cap would reject, accepted because it leans under 45 deg.
        solver, _, index = solver_scene
        pose = solver.solve_square(
            IN_BUDGET_TILTED_SQUARE,
            index.board,
            index.board_pose,
            np.random.default_rng(0),
            weight=0.0,
            max_approach_deg=MAX_TILT_DEG,
        )
        assert pose.is_within_tolerance()
        assert MAX_APPROACH_DEG < pose.approach_deg <= MAX_TILT_DEG


class TestPreferredTiltedSolutions:
    """The min-tilt search leaves roll free too, so a preference settles which branch it picks.

    Without one, consecutive anchors land in unrelated roll branches — a grasp near one roll stop
    while the survey pose sits near the other — and the joint-space legs between them spin the
    wrist the long way around. A preference pulls the chosen roll toward where the arm already is,
    which a pawn's axial symmetry makes free.
    """

    def _target(self, index: SceneIndex) -> npt.NDArray[np.float64]:
        return np.asarray(
            _above(index.board, PREFERENCE_SQUARE, index.board_pose, GRASP_HEIGHT),
            dtype=np.float64,
        )

    def test_a_preferred_tilted_solve_still_reaches_the_square(self, solver_scene: Scene) -> None:
        solver, _, index = solver_scene
        _, error, _, ok = solver.solve_min_tilt(
            self._target(index),
            np.random.default_rng(0),
            max_approach_deg=MAX_TILT_DEG,
            prefer=BIRDS_EYE_ARM_QPOS,
        )
        assert ok
        assert error <= MAX_TCP_ERROR

    def test_a_preference_turns_the_wrist_less(self, solver_scene: Scene) -> None:
        # Same restart set for both, so the preferred solve only reselects among the poses the
        # unpreferred one already found: it can never turn the wrist further from the survey roll.
        solver, _, index = solver_scene
        roll = ARM_JOINT_NAMES.index("wrist_roll")
        survey_roll = BIRDS_EYE_ARM_QPOS[roll]
        free, *_ = solver.solve_min_tilt(
            self._target(index), np.random.default_rng(0), max_approach_deg=MAX_TILT_DEG
        )
        biased, *_ = solver.solve_min_tilt(
            self._target(index),
            np.random.default_rng(0),
            max_approach_deg=MAX_TILT_DEG,
            prefer=BIRDS_EYE_ARM_QPOS,
        )
        assert abs(biased[roll] - survey_roll) < abs(free[roll] - survey_roll)
