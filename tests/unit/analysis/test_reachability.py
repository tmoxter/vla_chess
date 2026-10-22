import numpy as np
import pytest

from vla_chess.analysis.reachability import (
    MIN_SQUARE_CLEARANCE,
    STEP4_XY_JITTER,
    STEP4_YAW_JITTER_DEG,
    min_square_clearance,
    sample_board_pose,
)
from vla_chess.sim.scene import DEFAULT_BOARD_POSE, Board


def test_sample_board_pose_without_jitter_returns_the_nominal() -> None:
    pose = sample_board_pose(np.random.default_rng(0), xy_jitter=0.0, yaw_jitter_deg=0.0)
    assert pose == DEFAULT_BOARD_POSE


def test_sample_board_pose_stays_within_the_envelope() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        pose = sample_board_pose(rng)
        assert abs(pose.x - DEFAULT_BOARD_POSE.x) <= STEP4_XY_JITTER + 1e-9
        assert abs(pose.y - DEFAULT_BOARD_POSE.y) <= STEP4_XY_JITTER + 1e-9
        assert abs(np.degrees(pose.yaw - DEFAULT_BOARD_POSE.yaw)) <= STEP4_YAW_JITTER_DEG + 1e-9
        assert pose.square_pitch == DEFAULT_BOARD_POSE.square_pitch


def test_sample_board_pose_always_clears_the_base() -> None:
    board = Board()
    rng = np.random.default_rng(2)
    for _ in range(200):
        pose = sample_board_pose(rng)
        assert min_square_clearance(board, pose) >= MIN_SQUARE_CLEARANCE


def test_sample_board_pose_is_reproducible_from_its_seed() -> None:
    first = sample_board_pose(np.random.default_rng(3))
    second = sample_board_pose(np.random.default_rng(3))
    assert first == second


def test_sample_board_pose_actually_moves_the_board() -> None:
    poses = {sample_board_pose(np.random.default_rng(seed)) for seed in range(20)}
    # A real spread, not one fixed pose returned every time.
    assert len(poses) > 1


def test_sample_board_pose_rejects_negative_jitter() -> None:
    with pytest.raises(ValueError, match="xy_jitter"):
        sample_board_pose(np.random.default_rng(0), xy_jitter=-0.01)
    with pytest.raises(ValueError, match="yaw_jitter_deg"):
        sample_board_pose(np.random.default_rng(0), yaw_jitter_deg=-1.0)


def test_sample_board_pose_gives_up_when_no_draw_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every draw reported as inside the base: there is no clearing pose, so it must fail loudly
    # rather than return an overlapping board.
    monkeypatch.setattr(
        "vla_chess.analysis.reachability.min_square_clearance",
        lambda board, pose: MIN_SQUARE_CLEARANCE - 0.01,
    )
    with pytest.raises(RuntimeError, match="cleared the base"):
        sample_board_pose(np.random.default_rng(0), max_tries=5)
