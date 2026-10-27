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


@dataclass(frozen=True)
class TakeoverOutcome:
    """What one takeover produced: the situation it found, a kept episode or none, and the log."""

    situation: Situation
    episode: Episode | None
    attempts: int
    rejects: Counter[str] = field(default_factory=Counter)


def _reject_reason(result: ExecutionResult, target: str) -> str | None:
    """Name why a recovery run is not a keep, or None when it is one."""
    if result.abort_reason is not None:
        return result.abort_reason.value
    if result.final_square != target:
        return WRONG_SQUARE
    if not result.placed:
        return NOT_SETTLED
    return None


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
            dry = executor.run(restored(), plan, ik_abort=TILTED_IK_ABORT)
            reason = _reject_reason(dry, spec.target)
            if reason is None:
                kept = plan
                break
            rejects[reason] += 1

        if kept is None:
            return TakeoverOutcome(assessment.situation, None, attempts, rejects)

        with Renderer(model, self._cameras) as renderer:
            recorder = ObservationRecorder(renderer, fps=self._fps, control_hz=self._control_hz)
            result = executor.run(
                restored(),
                kept,
                on_control_step=recorder.on_control_step,
                ik_abort=TILTED_IK_ABORT,
            )
            frames = recorder.frames(result)
        if _reject_reason(result, spec.target) is not None or not frames:
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
    skip: int,
    max_attempts: int,
    rng: np.random.Generator,
    base_seed: int = 0,
) -> list[tuple[int, TakeoverOutcome]]:
    """Take one rollout over at random frames until enough recoveries are kept.

    Frames are tried in a random order. A frame whose move is finished or past saving is passed
    over without counting against the budget; every other one counts, kept or not.

    Args:
        rollout: The recorded rollout to take over.
        runner: What records a takeover from one snapshot.
        wanted: How many kept episodes to stop at.
        skip: How many leading frames never to take over from.
        max_attempts: How many recoverable frames to try at most.
        rng: Source of the frame order.
        base_seed: Offset of the per-frame seeds; frame `f` is taken over with
            `base_seed + f`.

    Returns:
        Every frame tried, with its outcome, in the order tried.
    """
    spec = rollout.spec()
    tried: list[tuple[int, TakeoverOutcome]] = []
    kept = attempts = 0
    for frame in takeover_order(len(rollout.snapshots), skip, rng):
        if kept >= wanted or attempts >= max_attempts:
            break
        outcome = runner.run(spec, rollout.snapshots[frame], base_seed=base_seed + frame)
        tried.append((frame, outcome))
        if outcome.situation in RECOVERABLE:
            attempts += 1
        if outcome.episode is not None:
            kept += 1
    return tried
