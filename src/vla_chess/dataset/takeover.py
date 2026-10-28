"""Expert takeover: record the expert finishing a move from a state another controller left behind.

The aggregation step of DAgger. A policy drives the arm and its rollout is kept as one simulator
snapshot per frame; at a chosen frame the scene is restored exactly, the expert reads where the
move stands, and it flies a recovery plan from there. Recoveries are judged by physics like the
funnel's candidates — each is dry-run unrendered, and the first that leaves the piece standing on
its target is re-run with the cameras attached and kept as an ordinary `Episode`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.control.executor import CONTROL_HZ, TILTED_IK_ABORT, ExecutionResult, Executor
from vla_chess.control.planner import ArmQpos, MoveSpec, Plan, PlanInfeasible
from vla_chess.control.recovery import (
    NotNearTarget,
    Situation,
    assess,
    plan_recovery,
    recovery_candidates,
)
from vla_chess.control.sampling import SampleKnobs
from vla_chess.dataset.episode import (
    ENGINE_VERSION_UNKNOWN,
    WRIST_CAMERA,
    Episode,
    move_scene_description,
)
from vla_chess.dataset.observe import DEFAULT_FPS, ObservationRecorder
from vla_chess.dataset.rollouts import RecordedRollout
from vla_chess.grasp.solver import VerticalGraspSolver
from vla_chess.sensor.render import CameraSpec, Renderer
from vla_chess.sim.collision import grip_contacts
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    BoardPose,
    PiecePlacement,
    build_scene,
    make_data,
)
from vla_chess.sim.snapshot import restore_state

# Why a recovery that ran was not kept, beyond the executor's own abort reasons.
WRONG_SQUARE = "wrong_square"
NOT_SETTLED = "not_settled"
PLAN_INFEASIBLE = "plan_infeasible"
NOT_NEAR_TARGET = "direct_not_near"
RELEASED_HIGH = "released_high"

# A piece let go of with its base higher than this above the playing surface was dropped rather
# than set down, in metres.
AIR_RELEASE = 0.003

# A piece let go of with its base higher than this above the playing surface was let go of too
# high, in metres. The clean expert sets a pawn down at a median of 2.6 mm (p90 3.5 mm, measured
# over 304 accepted funnel runs of the trained one-step moves), and 95% of its releases fall below
# this.
HIGH_RELEASE = 0.006

# How far the gripper command may rise above the tightest it held a piece at and still count as
# closed, in radians. A grip settles within a few thousandths of its command; an opening jaw clears
# this within a frame or two.
OPEN_DELTA = 0.01


@dataclass(frozen=True)
class TakeoverOutcome:
    """What one takeover produced: the situation it found, a kept episode or none, and the log."""

    situation: Situation
    episode: Episode | None
    attempts: int
    rejects: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class GripTrace:
    """Whether the moved piece was held, and how high it stood, at each frame of a rollout."""

    gripped: npt.NDArray[np.bool_]  # (frames,): both jaws bear on the piece
    height: npt.NDArray[np.float64]  # (frames,): piece base above the playing surface, m
    jaw_command: npt.NDArray[np.float64]  # (frames,): gripper joint command in effect, rad


@dataclass(frozen=True)
class Release:
    """One mid-air release of the moved piece, and whether the jaws were opened into it."""

    frame: int  # last frame the piece was held
    height: float  # piece base above the playing surface at that frame, m
    opened: bool  # the jaw command had risen past the grip by more than OPEN_DELTA
    last_closed: int  # last frame of the hold the jaws were still commanded closed


def mid_air_releases(trace: GripTrace, above: float) -> list[Release]:
    """Find where a rollout let go of its piece above the board, and how.

    A release counts as mid-air when the piece base stands higher than `above` on the last frame
    it was held. It was opened into when, by that frame, the jaw command had risen more than
    `OPEN_DELTA` above the tightest it reached during the hold; otherwise the piece slipped out
    of jaws still commanded closed.

    Args:
        trace: The rollout's grip trace.
        above: Height of the piece base above the playing surface, in metres, that a release
            has to exceed to count.

    Returns:
        The mid-air releases, in frame order.
    """
    releases = []
    for frame in np.flatnonzero(trace.gripped[:-1] & ~trace.gripped[1:]):
        if trace.height[frame] <= above:
            continue
        start = int(frame)
        while start > 0 and trace.gripped[start - 1]:
            start -= 1
        jaw = trace.jaw_command[start : frame + 1]
        closed = np.flatnonzero(jaw <= jaw.min() + OPEN_DELTA)
        last_closed = start + int(closed[-1])
        releases.append(
            Release(
                frame=int(frame),
                height=float(trace.height[frame]),
                opened=bool(last_closed < frame),
                last_closed=last_closed,
            )
        )
    return releases


def recovery_reject_reason(result: ExecutionResult, target: str, trace: GripTrace) -> str | None:
    """Name why a recovery run is not a keep, or None when it is one.

    A recovery is kept only when it ran to the end, left the piece standing on its target, and
    set it down rather than letting go of it higher than `HIGH_RELEASE`: one that lets go high
    demonstrates the release it is meant to correct, even when the piece happens to land upright.

    Args:
        result: The recovery's execution.
        target: The square the piece has to end on.
        trace: Grip trace of the recovery, one entry per control step.

    Returns:
        The executor's abort reason, `WRONG_SQUARE`, `NOT_SETTLED` or `RELEASED_HIGH`, in that
        order of precedence; None for a keep.
    """
    if result.abort_reason is not None:
        return result.abort_reason.value
    if result.final_square != target:
        return WRONG_SQUARE
    if not result.placed:
        return NOT_SETTLED
    if mid_air_releases(trace, HIGH_RELEASE):
        return RELEASED_HIGH
    return None


class _GripRecorder:
    """Builds a grip trace of a run as it executes, one entry per control step."""

    def __init__(
        self, model: mujoco.MjModel, piece_body: int, gripper_actuator: int, surface: float
    ) -> None:
        """Bind the recorder to the piece it watches.

        Args:
            model: Scene model the run executes in.
            piece_body: Body id of the moved piece.
            gripper_actuator: Actuator id of the gripper, whose command is read each step.
            surface: Height of the playing surface in the world frame, in metres.
        """
        self._model = model
        self._body = piece_body
        self._actuator = gripper_actuator
        self._surface = surface
        self._gripped: list[bool] = []
        self._height: list[float] = []
        self._jaw: list[float] = []

    def on_control_step(self, step: int, data: mujoco.MjData) -> None:
        """Record one stepped state; shaped for `Executor.run`'s callback."""
        self._gripped.append(grip_contacts(self._model, data, self._body).is_gripped())
        self._height.append(float(data.xpos[self._body][2]) - self._surface)
        self._jaw.append(float(data.ctrl[self._actuator]))

    def trace(self) -> GripTrace:
        """Return the trace recorded so far."""
        return GripTrace(
            gripped=np.array(self._gripped, dtype=bool),
            height=np.array(self._height),
            jaw_command=np.array(self._jaw),
        )


def release_order(trace: GripTrace, skip: int, lead: int, rng: np.random.Generator) -> list[int]:
    """Pick the frames of a rollout from before it opened the jaws on a piece held too high.

    For every release above `HIGH_RELEASE` the jaws were opened into, the last `lead` frames of
    the hold up to and including the last one still commanded closed are eligible; nothing else
    is. A rollout that slipped its piece, set it down, or never held it has no frames to try.

    Args:
        trace: The rollout's grip trace.
        skip: How many leading frames to leave out; the policy has barely acted on them.
        lead: How many held frames before the opening are eligible.
        rng: Source of the shuffle.

    Returns:
        The eligible frame indices in random order, for the caller to try until it has as many
        takeovers as it wants.
    """
    eligible: set[int] = set()
    for release in mid_air_releases(trace, HIGH_RELEASE):
        if not release.opened:
            continue
        first = release.last_closed - lead + 1
        while first < 0 or not trace.gripped[first]:
            first += 1
        eligible.update(range(max(first, skip), release.last_closed + 1))
    return [int(i) for i in rng.permutation(np.array(sorted(eligible), dtype=np.int64))]


def carry_order(
    trace: GripTrace, skip: int, lead: int, per_tier: int, rng: np.random.Generator
) -> list[int]:
    """Order the frames of a rollout so the ones carrying the piece are taken over first.

    Up to `per_tier` held frames from the last `lead` before a mid-air release (one with the
    piece base above `AIR_RELEASE`) come first, then up to `per_tier` other held frames, then
    every remaining frame in a random order. The cap keeps a rollout whose grip cannot be saved
    from spending its whole budget on held frames.

    Args:
        trace: The rollout's grip trace.
        skip: How many leading frames to leave out; the policy has barely acted on them.
        lead: How many held frames before a mid-air release are eligible for the first tier.
        per_tier: How many frames each of the two held tiers puts ahead of the rest.
        rng: Source of the shuffles.

    Returns:
        Every frame index from `skip` on, for the caller to try until it has as many takeovers
        as it wants.
    """
    frames = len(trace.gripped)
    candidates = np.arange(min(skip, frames), frames)
    held = trace.gripped[candidates]
    before_drop = np.zeros(frames, dtype=bool)
    releases = np.flatnonzero(trace.gripped[:-1] & ~trace.gripped[1:])
    for release in releases:
        if trace.height[release] > AIR_RELEASE:
            before_drop[max(0, release - lead + 1) : release + 1] = True
    first = rng.permutation(candidates[held & before_drop[candidates]])[:per_tier]
    second = rng.permutation(candidates[held & ~before_drop[candidates]])[:per_tier]
    rest = rng.permutation(np.setdiff1d(candidates, np.concatenate([first, second])))
    return [int(i) for tier in (first, second, rest) for i in tier]


def takeover_order(frames: int, skip: int, rng: np.random.Generator) -> list[int]:
    """Shuffle the frames of a rollout that a takeover may start from.

    Args:
        frames: How many frames the rollout has.
        skip: How many leading frames to leave out; the policy has barely acted on them.
        rng: Source of the shuffle.

    Returns:
        Every frame index from `skip` on, in random order, for the caller to try until it has
        as many takeovers as it wants.
    """
    return [int(i) for i in rng.permutation(np.arange(min(skip, frames), frames))]


class TakeoverRunner:
    """Records the expert finishing moves from snapshots, on one fixed scene layout.

    The layout — pieces, board pose and survey pose — has to be the one the snapshots were taken
    in, since a snapshot only restores into the model it came from.
    """

    def __init__(
        self,
        *,
        pieces: Sequence[PiecePlacement],
        fps: float = DEFAULT_FPS,
        control_hz: float = CONTROL_HZ,
        board_pose: BoardPose = DEFAULT_BOARD_POSE,
        survey_qpos: ArmQpos = BIRDS_EYE_ARM_QPOS,
        cameras: Sequence[CameraSpec] = (WRIST_CAMERA,),
        knobs: SampleKnobs | None = None,
        engine_version: str = ENGINE_VERSION_UNKNOWN,
    ) -> None:
        """Fix the scene layout and capture settings every takeover of this runner shares.

        Args:
            pieces: The board layout the snapshots were taken on.
            fps: Dataset frame rate the observations are captured at.
            control_hz: Rate the control loop writes commands and the recorder samples from.
            board_pose: Where the board stood in the world.
            survey_qpos: Arm configuration the scene was built resting at.
            cameras: The views to render; the wrist camera the dataset carries is the default.
            knobs: The grasp lean and placement offset spread recoveries are drawn from; the
                funnel's defaults when omitted.
            engine_version: Identifier of the engine, stamped onto each kept episode.
        """
        self._pieces = tuple(pieces)
        self._fps = fps
        self._control_hz = control_hz
        self._board_pose = board_pose
        self._survey_qpos = survey_qpos
        self._cameras = tuple(cameras)
        self._knobs = knobs if knobs is not None else SampleKnobs()
        self._engine_version = engine_version

    def run(
        self, spec: MoveSpec, snapshot: npt.NDArray[np.float64], *, base_seed: int = 0
    ) -> TakeoverOutcome:
        """Take a move over from one snapshot and record the expert finishing it.

        Args:
            spec: The move that was being made when the snapshot was taken.
            snapshot: The state to take over from, from `sim.snapshot.capture_state` in a scene
                of this runner's layout built for `spec`.
            base_seed: Seed of the recoveries' restart searches and lean spread.

        Returns:
            The situation the snapshot was in, the recorded episode when a recovery placed the
            piece (None otherwise), how many recoveries were executed, and why the others failed.
        """
        description = move_scene_description(
            spec, self._pieces, self._board_pose, self._survey_qpos
        )
        model, index = build_scene(description)

        def restored() -> mujoco.MjData:
            data = make_data(model, index)
            restore_state(model, data, snapshot)
            return data

        start = restored()
        scratch = make_data(model, index)
        assessment = assess(model, start, index, spec, VerticalGraspSolver(model, scratch, index))
        executor = Executor(model, index, control_hz=self._control_hz)
        body = index.piece_body_id[spec.source]
        surface = index.board.surface_height

        def grip_recorder() -> _GripRecorder:
            return _GripRecorder(model, body, index.gripper_actuator_id, surface)

        rejects: Counter[str] = Counter()
        attempts = 0
        kept: Plan | None = None
        for candidate in recovery_candidates(spec, assessment.situation, self._knobs, base_seed):
            plan_data = make_data(model, index)
            try:
                plan = plan_recovery(
                    candidate,
                    assessment,
                    model,
                    plan_data,
                    index,
                    VerticalGraspSolver(model, plan_data, index),
                    self._knobs.max_tilt_deg,
                )
            except NotNearTarget:
                rejects[NOT_NEAR_TARGET] += 1
                continue
            except PlanInfeasible:
                rejects[PLAN_INFEASIBLE] += 1
                continue
            attempts += 1
            watch = grip_recorder()
            dry = executor.run(
                restored(), plan, on_control_step=watch.on_control_step, ik_abort=TILTED_IK_ABORT
            )
            reason = recovery_reject_reason(dry, spec.target, watch.trace())
            if reason is None:
                kept = plan
                break
            rejects[reason] += 1

        if kept is None:
            return TakeoverOutcome(assessment.situation, None, attempts, rejects)

        with Renderer(model, self._cameras) as renderer:
            recorder = ObservationRecorder(renderer, fps=self._fps, control_hz=self._control_hz)
            watch = grip_recorder()

            def on_control_step(step: int, data: mujoco.MjData) -> None:
                recorder.on_control_step(step, data)
                watch.on_control_step(step, data)

            result = executor.run(
                restored(),
                kept,
                on_control_step=on_control_step,
                ik_abort=TILTED_IK_ABORT,
            )
            frames = recorder.frames(result)
        if recovery_reject_reason(result, spec.target, watch.trace()) is not None or not frames:
            # The rendered re-run must reproduce the dry run it was chosen from.
            rejects["not_reproduced"] += 1
            return TakeoverOutcome(assessment.situation, None, attempts, rejects)
        episode = Episode(
            spec=spec,
            instruction=kept.instruction,
            frames=frames,
            execution=result,
            engine_version=self._engine_version,
            fps=self._fps,
        )
        return TakeoverOutcome(assessment.situation, episode, attempts, rejects)

    def grip_trace(self, spec: MoveSpec, snapshots: npt.NDArray[np.float64]) -> GripTrace:
        """Read whether the moved piece was held, and its height, at every snapshot of a rollout.

        Args:
            spec: The move the rollout was making.
            snapshots: One state per frame, from `sim.snapshot.capture_state` in a scene of this
                runner's layout built for `spec`.

        Returns:
            The rollout's grip trace.
        """
        description = move_scene_description(
            spec, self._pieces, self._board_pose, self._survey_qpos
        )
        model, index = build_scene(description)
        data = make_data(model, index)
        body = index.piece_body_id[spec.source]
        surface = index.board.surface_height
        gripped = np.zeros(len(snapshots), dtype=bool)
        height = np.zeros(len(snapshots))
        jaw_command = np.zeros(len(snapshots))
        for frame, snapshot in enumerate(snapshots):
            restore_state(model, data, snapshot)
            gripped[frame] = grip_contacts(model, data, body).is_gripped()
            height[frame] = float(data.xpos[body][2]) - surface
            jaw_command[frame] = float(data.ctrl[index.gripper_actuator_id])
        return GripTrace(gripped=gripped, height=height, jaw_command=jaw_command)


class Takeover(Protocol):
    """The part of `TakeoverRunner` the rollout loop needs."""

    def run(
        self, spec: MoveSpec, snapshot: npt.NDArray[np.float64], *, base_seed: int = 0
    ) -> TakeoverOutcome:
        """Take a move over from one snapshot; see `TakeoverRunner.run`."""
        ...


# The situations a takeover can do something about; the others cost a scene build and nothing more.
RECOVERABLE: frozenset[Situation] = frozenset(
    {Situation.TO_GRASP, Situation.IN_HAND, Situation.PLACED}
)


def take_over_rollout(
    rollout: RecordedRollout,
    runner: Takeover,
    *,
    wanted: int,
    max_attempts: int,
    order: Sequence[int],
    base_seed: int = 0,
) -> list[tuple[int, TakeoverOutcome]]:
    """Take one rollout over frame by frame until enough recoveries are kept.

    A frame whose move is finished or past saving is passed over without counting against the
    budget; every other one counts, kept or not.

    Args:
        rollout: The recorded rollout to take over.
        runner: What records a takeover from one snapshot.
        wanted: How many kept episodes to stop at.
        max_attempts: How many recoverable frames to try at most.
        order: The frames to try, in the order to try them; see `takeover_order`,
            `carry_order` and `release_order`.
        base_seed: Offset of the per-frame seeds; frame `f` is taken over with
            `base_seed + f`.

    Returns:
        Every frame tried, with its outcome, in the order tried.
    """
    spec = rollout.spec()
    tried: list[tuple[int, TakeoverOutcome]] = []
    kept = attempts = 0
    for frame in order:
        if kept >= wanted or attempts >= max_attempts:
            break
        outcome = runner.run(spec, rollout.snapshots[frame], base_seed=base_seed + frame)
        tried.append((frame, outcome))
        if outcome.situation in RECOVERABLE:
            attempts += 1
        if outcome.episode is not None:
            kept += 1
    return tried
