"""Per-frame observation capture: which control steps become dataset frames, and what they carry."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Final, Protocol

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.control.executor import ExecutionResult
from vla_chess.control.planner import Phase
from vla_chess.sensor.render import RgbFrame

DEFAULT_FPS: Final = 10.0

# The buffer is capped so a runaway episode raises rather than growing until the process dies. The
# ceiling is a *duration*, not a frame count: the historical 400-frame cap was 40 s at the 10 fps
# baseline (docs/render-budget.md, measured on a 4 GB host). Capping by frame count alone silently
# clips a faster capture to a fraction of an episode — a 30 fps run tops out at ~13 s and raises
# mid-episode — so the default scales with fps to hold the same 40 s ceiling (1200 frames @ 30 fps).
MAX_BUFFERED_SECONDS: Final = 40.0
MAX_BUFFERED_FRAMES: Final = 400  # == MAX_BUFFERED_SECONDS * DEFAULT_FPS; kept for callers/tests


class FrameSource(Protocol):
    """The part of `render.Renderer` this module needs."""

    def render(self, data: mujoco.MjData) -> dict[str, RgbFrame]:
        """Render every camera for one simulated state.

        Args:
            data: State to view.

        Returns:
            One frame per camera, keyed by camera name.
        """
        ...


@dataclass(frozen=True)
class Frame:
    """One dataset frame: what the camera saw, the state then, and the action taken from it."""

    images: dict[str, RgbFrame]
    state: npt.NDArray[np.float64]
    action: npt.NDArray[np.float64]
    timestamp: float
    frame_index: int
    control_step: int
    phase: Phase


class ObservationRecorder:
    """Captures images at the dataset frame rate while an episode is executing.

    Images only: state, action and phase are read back from the `ExecutionResult` in `frames`,
    so the action-alignment convention lives in one place rather than being reconstructed from
    whatever the control loop happened to have in hand at capture time.
    """

    def __init__(
        self,
        renderer: FrameSource,
        fps: float = DEFAULT_FPS,
        control_hz: float = 50.0,
        max_buffered_frames: int | None = None,
    ) -> None:
        """Set the frame clock this recorder captures on.

        Args:
            renderer: What to capture through, called once per captured step.
            fps: Dataset frame rate in hertz, which is what the frame clock counts in.
            control_hz: Rate the control loop calls the hook at, used to turn a step index
                into a time.
            max_buffered_frames: How many frames may be held in memory before capture refuses
                to grow further. Defaults to a 40 s ceiling scaled to `fps` (so 30 fps allows
                1200 frames, not the 10 fps baseline of 400), never below the 400-frame baseline.

        Raises:
            ValueError: If either rate is not positive, or the capture rate exceeds the
                control rate it would have to sample from.
        """
        if fps <= 0.0 or control_hz <= 0.0:
            raise ValueError(f"fps and control_hz must be positive, got {fps} and {control_hz}")
        if fps > control_hz:
            raise ValueError(
                f"cannot capture {fps} frames per second from a {control_hz} Hz control loop"
            )
        self._renderer = renderer
        self._fps = fps
        self._control_hz = control_hz
        if max_buffered_frames is None:
            # Duration ceiling scaled to the capture rate; never below the 10 fps baseline.
            max_buffered_frames = max(MAX_BUFFERED_FRAMES, ceil(MAX_BUFFERED_SECONDS * fps))
        self._max_buffered = max_buffered_frames
        self._captured: list[tuple[int, dict[str, RgbFrame]]] = []
        self._next_capture = 0.0

    @property
    def fps(self) -> float:
        """Report the rate frames are captured at.

        Returns:
            The dataset frame rate in hertz.
        """
        return self._fps

    @property
    def captured_steps(self) -> tuple[int, ...]:
        """List the control steps that became frames.

        Returns:
            Their step indices, in capture order.
        """
        return tuple(step for step, _ in self._captured)

    def on_control_step(self, step: int, data: mujoco.MjData) -> None:
        """Render this control step if the frame clock has reached it.

        Args:
            step: Index of the control step just taken.
            data: State as it stands after that step.

        Raises:
            RuntimeError: If the buffer already holds its maximum number of frames, rather
                than growing until the process runs out of memory.
        """
        if step / self._control_hz < self._next_capture - 1e-9:
            return
        if len(self._captured) >= self._max_buffered:
            raise RuntimeError(
                f"observation buffer is full at {self._max_buffered} frames; either shorten the "
                "episode, lower the capture rate, or raise max_buffered_frames deliberately"
            )
        rendered = self._renderer.render(data)
        # Copied: the renderer hands back views of buffers it overwrites on the next call.
        self._captured.append(
            (step, {name: np.array(image, copy=True) for name, image in rendered.items()})
        )
        self._next_capture += 1.0 / self._fps

    def frames(self, result: ExecutionResult) -> list[Frame]:
        """Pair the captured images with the states and actions the run recorded.

        Frame `i` carries the observation at its own control step and the action that moves the
        arm on from it — the executor's action at step `s` is the command that produced state
        `s`, so the dataset action for a frame at step `s` is the executor's action at `s + 1`.
        The last frame repeats its own command, having no successor to point at.

        Args:
            result: The finished rollout to read states, actions and phases from. Captures
                past its last recorded step are dropped, which is what an aborted run leaves
                behind.

        Returns:
            One frame per surviving capture, in order.
        """
        frames = []
        for frame_index, (step, images) in enumerate(self._captured):
            if step >= len(result.states):
                break
            following = min(step + 1, len(result.actions) - 1)
            frames.append(
                Frame(
                    images=images,
                    state=result.states[step],
                    action=result.actions[following],
                    timestamp=frame_index / self._fps,
                    frame_index=frame_index,
                    control_step=step,
                    phase=result.phases[step],
                )
            )
        return frames


def image_is_sane(image: RgbFrame) -> bool:
    """Judge whether a frame looks like something was actually rendered.

    Args:
        image: One rendered frame, as (height, width, 3) uint8.

    Returns:
        True when the frame has enough spread to carry detail and sits clear of both black
        and blown-out — the two ways an empty render shows up.
    """
    return float(np.std(image)) > 5.0 and 10.0 <= float(np.mean(image)) <= 245.0
