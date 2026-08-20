import itertools

import numpy as np
import numpy.typing as npt
import pytest

from vla_chess.control.executor import ExecutionDiagnostics, ExecutionResult
from vla_chess.control.planner import Phase
from vla_chess.dataset.observe import Frame, ObservationRecorder, image_is_sane


class StubRenderer:
    """Hands back a distinct constant frame per call, so frames can be told apart."""

    def __init__(self, size: int = 8) -> None:
        self.calls = 0
        self._size = size

    def render(self, data: object) -> dict[str, npt.NDArray[np.uint8]]:
        self.calls += 1
        return {"wrist": np.full((self._size, self._size, 3), self.calls % 256, dtype=np.uint8)}


def synthetic_result(steps: int) -> ExecutionResult:
    """A rollout where state[t] = t and action[t] = t, so alignment is readable off the values."""
    states = np.tile(np.arange(steps, dtype=np.float64).reshape(-1, 1), (1, 6))
    return ExecutionResult(
        states=states,
        actions=states.copy(),
        phases=tuple(Phase.TRANSPORT for _ in range(steps)),
        phase_boundaries={Phase.TRANSPORT: (0, steps - 1)},
        grasped=True,
        placed=True,
        final_square="d6",
        abort_reason=None,
        diagnostics=ExecutionDiagnostics(0.0, 0.0, steps, steps, 0.0),
    )


def record(steps: int, fps: float, control_hz: float) -> tuple[ObservationRecorder, list[Frame]]:
    recorder = ObservationRecorder(StubRenderer(), fps=fps, control_hz=control_hz)
    for step in range(steps):
        recorder.on_control_step(step, None)
    return recorder, recorder.frames(synthetic_result(steps))


def test_capture_rate_produces_the_expected_frame_count() -> None:
    _, frames = record(steps=500, fps=15.0, control_hz=50.0)
    assert len(frames) == round(10.0 * 15.0)


def test_timestamps_are_monotonic_at_the_capture_period() -> None:
    _, frames = record(steps=500, fps=15.0, control_hz=50.0)
    gaps = np.diff([frame.timestamp for frame in frames])
    assert np.all(np.abs(gaps - 1.0 / 15.0) < 1e-6)


def test_frame_indices_are_contiguous_from_zero() -> None:
    _, frames = record(steps=200, fps=10.0, control_hz=50.0)
    assert [frame.frame_index for frame in frames] == list(range(len(frames)))


def test_the_action_on_a_frame_is_the_one_that_reaches_the_next_state() -> None:
    _, frames = record(steps=50, fps=50.0, control_hz=50.0)
    for frame, following in itertools.pairwise(frames):
        assert frame.action[0] == pytest.approx(following.state[0])


def test_the_last_frame_repeats_its_own_command() -> None:
    _, frames = record(steps=50, fps=50.0, control_hz=50.0)
    assert frames[-1].action[0] == pytest.approx(frames[-1].state[0])


def test_every_frame_carries_the_wrist_camera() -> None:
    _, frames = record(steps=100, fps=10.0, control_hz=50.0)
    for frame in frames:
        assert frame.images["wrist"].dtype == np.uint8
        assert frame.images["wrist"].shape == (8, 8, 3)


def test_frames_are_copies_not_views_of_a_reused_buffer() -> None:
    _, frames = record(steps=100, fps=10.0, control_hz=50.0)
    assert not np.array_equal(frames[0].images["wrist"], frames[1].images["wrist"])


def test_capture_cannot_outrun_the_control_loop() -> None:
    with pytest.raises(ValueError, match="cannot capture"):
        ObservationRecorder(StubRenderer(), fps=100.0, control_hz=50.0)


def test_the_buffer_raises_rather_than_growing_without_bound() -> None:
    recorder = ObservationRecorder(StubRenderer(), fps=50.0, control_hz=50.0, max_buffered_frames=3)
    for step in range(3):
        recorder.on_control_step(step, None)
    with pytest.raises(RuntimeError, match="buffer is full"):
        recorder.on_control_step(3, None)


def test_image_sanity_rejects_blank_frames() -> None:
    assert not image_is_sane(np.zeros((8, 8, 3), dtype=np.uint8))
    assert not image_is_sane(np.full((8, 8, 3), 255, dtype=np.uint8))
    assert image_is_sane(np.random.default_rng(0).integers(0, 255, size=(8, 8, 3), dtype=np.uint8))
