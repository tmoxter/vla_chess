"""When a move is done: the arm has let go of the piece and come back to rest.

A move has no fixed length once a policy flies it, so its end has to be read off the arm. The rule
here needs only the arm's joint angles and whether the hand is gripping, both of which a real arm
reports: the hand has to have held something for a while and let go of it, and the arm then has to
stand at its rest pose, not gripping, for a run of frames. Resting before anything was gripped, as
the arm does at the start of every move, does not count.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from vla_chess.sim.scene import BIRDS_EYE_ARM_QPOS

# How far any arm joint may stand from the rest pose and still count as at rest, in radians. A
# policy that flies home stops within about 0.01 rad of it; the expert's own recovery counts the
# arm home at 0.05 rad.
DONE_REST_TOLERANCE: Final = 0.1

# Frames the arm has to stand at rest, not gripping, before the move counts as done: half a second
# at the dataset's 30 fps.
DONE_HOLD_FRAMES: Final = 15

# Consecutive gripping frames that arm the rule; a brush of the jaws against the piece is shorter.
DONE_GRIP_FRAMES: Final = 5


class DoneDetector:
    """Reads, one frame at a time, whether a move is done.

    Once it has fired it stays fired until `reset`.
    """

    def __init__(
        self,
        rest_qpos: Sequence[float] | npt.NDArray[np.floating] = BIRDS_EYE_ARM_QPOS,
        tolerance: float = DONE_REST_TOLERANCE,
        hold_frames: int = DONE_HOLD_FRAMES,
        grip_frames: int = DONE_GRIP_FRAMES,
    ) -> None:
        """Fix the rest pose and the rule's thresholds.

        Args:
            rest_qpos: The arm's rest pose, five joint angles in radians.
            tolerance: Largest joint distance from the rest pose that counts as at rest, radians.
            hold_frames: Frames at rest, not gripping, that make the move done.
            grip_frames: Consecutive gripping frames before the rule can fire at all.

        Raises:
            ValueError: If the tolerance is not positive or either frame count is below one.
        """
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        if hold_frames < 1 or grip_frames < 1:
            raise ValueError(
                f"hold_frames and grip_frames must be at least 1, got {hold_frames}, {grip_frames}"
            )
        self._rest = np.asarray(rest_qpos, dtype=np.float64)
        self._tolerance = tolerance
        self._hold_frames = hold_frames
        self._grip_frames = grip_frames
        self._frame = 0
        self._gripping_run = 0
        self._resting_run = 0
        self._armed = False
        self._done_frame: int | None = None

    @property
    def done_frame(self) -> int | None:
        """Index of the frame the move became done on, or None while it is not done."""
        return self._done_frame

    def reset(self) -> None:
        """Forget everything seen, for a new move."""
        self._frame = 0
        self._gripping_run = 0
        self._resting_run = 0
        self._armed = False
        self._done_frame = None

    def update(self, arm_qpos: Sequence[float] | npt.NDArray[np.floating], gripped: bool) -> bool:
        """Take in one frame and report whether the move is done.

        Args:
            arm_qpos: The five arm joint angles at this frame, radians.
            gripped: Whether the hand grips something at this frame.

        Returns:
            True from the frame the move became done on.
        """
        frame = self._frame
        self._frame += 1
        if self._done_frame is not None:
            return True
        self._gripping_run = self._gripping_run + 1 if gripped else 0
        if self._gripping_run >= self._grip_frames:
            self._armed = True
        distance = float(np.max(np.abs(np.asarray(arm_qpos, dtype=np.float64) - self._rest)))
        if self._armed and not gripped and distance < self._tolerance:
            self._resting_run += 1
        else:
            self._resting_run = 0
        if self._resting_run >= self._hold_frames:
            self._done_frame = frame
            return True
        return False


def done_frame(
    arm_qpos: npt.NDArray[np.floating],
    gripped: npt.NDArray[np.bool_],
    rest_qpos: Sequence[float] | npt.NDArray[np.floating] = BIRDS_EYE_ARM_QPOS,
    tolerance: float = DONE_REST_TOLERANCE,
    hold_frames: int = DONE_HOLD_FRAMES,
    grip_frames: int = DONE_GRIP_FRAMES,
) -> int | None:
    """Find the frame a recorded move became done on.

    Args:
        arm_qpos: Arm joint angles per frame, shape (frames, 5), radians.
        gripped: Whether the hand gripped something, per frame.
        rest_qpos: The arm's rest pose, radians.
        tolerance: Largest joint distance from the rest pose that counts as at rest, radians.
        hold_frames: Frames at rest, not gripping, that make the move done.
        grip_frames: Consecutive gripping frames before the rule can fire at all.

    Returns:
        The index of the frame the move became done on, or None if it never did.
    """
    detector = DoneDetector(rest_qpos, tolerance, hold_frames, grip_frames)
    for qpos, held in zip(arm_qpos, gripped, strict=True):
        if detector.update(qpos, bool(held)):
            return detector.done_frame
    return None
