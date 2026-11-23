import numpy as np
import pytest

from vla_chess.control.done import DoneDetector, done_frame

REST = np.array([0.1, -1.2, 0.3, 1.5, -1.5])
AWAY = REST + np.array([0.0, 0.6, -0.4, 0.0, 0.0])


def _detector() -> DoneDetector:
    return DoneDetector(rest_qpos=REST, tolerance=0.1, hold_frames=3, grip_frames=2)


def _feed(detector: DoneDetector, frames: list[tuple[np.ndarray, bool]]) -> list[bool]:
    return [detector.update(qpos, gripped) for qpos, gripped in frames]


def test_resting_before_any_grip_is_not_done() -> None:
    detector = _detector()
    assert not any(_feed(detector, [(REST, False)] * 10))
    assert detector.done_frame is None


def test_a_grip_released_and_a_return_to_rest_is_done_once_held_long_enough() -> None:
    detector = _detector()
    frames = [(AWAY, True)] * 2 + [(AWAY, False)] * 2 + [(REST, False)] * 4
    assert _feed(detector, frames) == [False] * 6 + [True, True]
    assert detector.done_frame == 6


def test_a_grip_shorter_than_asked_for_does_not_arm_the_detector() -> None:
    detector = _detector()
    frames = [(AWAY, True)] + [(AWAY, False)] + [(REST, False)] * 5
    assert not any(_feed(detector, frames))


def test_leaving_rest_before_the_hold_is_up_restarts_the_count() -> None:
    detector = _detector()
    frames = (
        [(AWAY, True)] * 2
        + [(REST, False)] * 2
        + [(AWAY, False)]
        + [(REST, False)] * 2
        + [(REST, False)]
    )
    assert _feed(detector, frames) == [False] * 7 + [True]


def test_gripping_at_rest_does_not_count_towards_the_hold() -> None:
    detector = _detector()
    frames = [(AWAY, True)] * 2 + [(REST, True)] * 5
    assert not any(_feed(detector, frames))


def test_a_pose_just_inside_the_tolerance_counts_as_rest() -> None:
    detector = _detector()
    near = REST + np.array([0.0, 0.0, 0.0, 0.099, -0.099])
    frames = [(AWAY, True)] * 2 + [(near, False)] * 3
    assert _feed(detector, frames)[-1]


def test_done_stays_latched_once_fired() -> None:
    detector = _detector()
    _feed(detector, [(AWAY, True)] * 2 + [(REST, False)] * 3)
    assert detector.update(AWAY, True)
    assert detector.done_frame == 4


def test_reset_forgets_the_previous_episode() -> None:
    detector = _detector()
    _feed(detector, [(AWAY, True)] * 2 + [(REST, False)] * 3)
    detector.reset()
    assert detector.done_frame is None
    assert not any(_feed(detector, [(REST, False)] * 5))


def test_done_frame_reads_a_whole_trace() -> None:
    arm = np.stack([AWAY, AWAY, AWAY, REST, REST, REST, REST])
    gripped = np.array([True, True, False, False, False, False, False])
    assert (
        done_frame(arm, gripped, rest_qpos=REST, tolerance=0.1, hold_frames=3, grip_frames=2) == 5
    )


def test_done_frame_of_a_trace_that_never_finishes_is_none() -> None:
    arm = np.stack([AWAY] * 5)
    gripped = np.array([True, True, False, False, False])
    assert done_frame(arm, gripped, rest_qpos=REST) is None


@pytest.mark.parametrize(
    "kwargs",
    [{"tolerance": 0.0}, {"hold_frames": 0}, {"grip_frames": 0}],
)
def test_the_rule_rejects_degenerate_settings(kwargs: dict[str, float]) -> None:
    settings = {"tolerance": 0.1, "hold_frames": 3, "grip_frames": 2, **kwargs}
    with pytest.raises(ValueError):
        DoneDetector(rest_qpos=REST, **settings)  # type: ignore[arg-type]
