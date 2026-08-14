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

# Squares the arm can reach fingers-down but never without colliding with itself; see
# docs/grasp-poses.md.
BLOCKED_HOME_SQUARES = {"d8", "e8"}


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
        needed = {f"{file}{rank}" for file in "abcdefgh" for rank in "678"}
        assert needed - BLOCKED_HOME_SQUARES <= set(grasps)

    def test_the_two_blocked_home_squares_are_still_blocked(
        self, grasps: dict[str, GraspPose]
    ) -> None:
        # d8 and e8 sit directly in front of the shoulder, and every fingers-down solution for
        # them folds the arm through its own body — not a solver budget problem: a 60-restart
        # search found 186 in-tolerance poses for each and no clean one. Pinned here because
        # the gap is load-bearing for move planning (the black queen and king cannot be lifted
        # off their home squares this way) and because it is exactly the kind of thing that
        # gets quietly reintroduced by a solver change that stops screening collisions.
        assert BLOCKED_HOME_SQUARES.isdisjoint(grasps)

    def test_unreachable_squares_are_absent_rather_than_recorded(self) -> None:
        payload = json.loads(REFERENCE_GRASPS_PATH.read_text())
        assert "a1" not in payload["grasps"]
