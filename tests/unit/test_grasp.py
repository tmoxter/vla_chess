import json

import pytest

from vla_chess.grasp import (
    GRASP_HEIGHT,
    MAX_APPROACH_DEG,
    MAX_TCP_ERROR,
    REFERENCE_BOARD_POSE,
    REFERENCE_GRASPS_PATH,
    GraspPose,
    load_reference_grasps,
    reference_metadata,
)
from vla_chess.scene import ARM_JOINT_NAMES, DEFAULT_BOARD_POSE

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
