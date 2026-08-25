"""The generation engine's entry point: a MoveSpec in, a recorded Episode out."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from vla_chess.control.executor import CONTROL_HZ, ExecutionResult, Executor
from vla_chess.control.planner import ArmQpos, MoveSpec, plan_move
from vla_chess.dataset.observe import DEFAULT_FPS, Frame, ObservationRecorder
from vla_chess.grasp.solver import VerticalGraspSolver
from vla_chess.sensor.render import CameraSpec, Renderer
from vla_chess.sim.positions import starting_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    RENDER_OFFHEIGHT,
    RENDER_OFFWIDTH,
    WRIST_CAMERA_NAME,
    BoardPose,
    PiecePlacement,
    SceneDescription,
    build_scene,
    grip_radius,
    make_data,
    pinch_offset_for,
)

ENGINE_VERSION_UNKNOWN: Final = "unknown"

# The one view the policy is trained on. The witness view the single-episode CLI also renders is
# for humans looking at a run, not for the dataset, so the runner does not carry it by default.
WRIST_CAMERA: Final = CameraSpec(WRIST_CAMERA_NAME, RENDER_OFFWIDTH, RENDER_OFFHEIGHT)


@dataclass(frozen=True)
class Episode:
    """One recorded pick-and-place run: the move, its frames, and the label attached to it."""

    spec: MoveSpec
    instruction: str
    frames: list[Frame]
    execution: ExecutionResult
    engine_version: str

    @property
    def succeeded(self) -> bool:
        """Report whether the run gripped the piece and left it on its destination.

        Returns:
            The underlying rollout's own verdict; a failed episode is still a complete
            `Episode`, since the frames of a failure are worth keeping.
        """
        return self.execution.succeeded


def move_scene_description(
    spec: MoveSpec,
    pieces: Sequence[PiecePlacement],
    board_pose: BoardPose,
    survey_qpos: ArmQpos,
) -> SceneDescription:
    """Assemble the scene description one move is planned and executed against.

    The pinch site is built for the piece this move lifts, so the piece ends up centred
    between the jaws rather than shoved against one face.

    Args:
        spec: The move to run; its source square must carry a piece of `spec.piece_type`.
        pieces: The board layout to build, one placement per occupied square.
        board_pose: Where the board stands in the world.
        survey_qpos: Arm configuration the scene starts and rests at between legs.

    Returns:
        The description `build_scene` assembles the move's model from.

    Raises:
        ValueError: If the source square is empty in `pieces`, or the piece standing on it is
            not the type the spec names.
    """
    occupant = {placement.square: placement.piece_type for placement in pieces}
    standing = occupant.get(spec.source)
    if standing is None:
        raise ValueError(f"no piece on {spec.source} to move")
    if standing != spec.piece_type:
        raise ValueError(
            f"{spec.source} carries a {standing.value}, but the move "
            f"names a {spec.piece_type.value}"
        )
    return SceneDescription(
        board_pose=board_pose,
        pieces=tuple(pieces),
        arm_qpos=survey_qpos,
        pinch_offset=pinch_offset_for(grip_radius(spec.piece_type, spec.grasp_height)),
    )


class EpisodeRunner:
    """Turns moves into recorded episodes, one scene per move, on a fixed board layout.

    The board pose, piece layout and survey pose are the runner's, fixed across the moves it is
    given; the move — which piece goes where — is the per-episode input. Every move rebuilds its
    own model rather than mutating a shared one, so no state leaks from one episode into the
    next; scene-level reuse is a throughput concern left to the fuller engine.
    """

    def __init__(
        self,
        *,
        fps: float = DEFAULT_FPS,
        control_hz: float = CONTROL_HZ,
        pieces: Sequence[PiecePlacement] | None = None,
        board_pose: BoardPose = DEFAULT_BOARD_POSE,
        survey_qpos: ArmQpos = BIRDS_EYE_ARM_QPOS,
        cameras: Sequence[CameraSpec] = (WRIST_CAMERA,),
        engine_version: str = ENGINE_VERSION_UNKNOWN,
    ) -> None:
        """Fix the board and capture settings every episode this runner produces shares.

        Args:
            fps: Dataset frame rate the observations are captured at.
            control_hz: Rate the control loop writes commands and the recorder samples from.
            pieces: The board layout to run moves on. The standard opening is used when
                omitted.
            board_pose: Where the board stands in the world.
            survey_qpos: Arm configuration the scene starts and rests at between legs.
            cameras: The views to render; the wrist camera the dataset carries is the default.
            engine_version: Identifier of the engine that produced the episodes, stamped onto
                each. Pass the git SHA (see `git_engine_version`) so a dataset can be traced
                back to the code that generated it.
        """
        self._fps = fps
        self._control_hz = control_hz
        self._pieces = tuple(pieces) if pieces is not None else starting_position()
        self._board_pose = board_pose
        self._survey_qpos = survey_qpos
        self._cameras = tuple(cameras)
        self._engine_version = engine_version

    def run(self, spec: MoveSpec, *, seed: int = 0) -> Episode:
        """Plan, execute and record one move.

        Args:
            spec: The move to run, on the runner's board layout.
            seed: Seed for the grasp solver's random restarts. Fixed rather than varied — the
                slice authors near-identical demonstrations on purpose — but distinct seeds
                across a batch keep independent moves from sharing a restart sequence.

        Returns:
            The recorded episode, whether or not the piece was placed. Read
            `Episode.succeeded` for the verdict.

        Raises:
            ValueError: If the source square does not carry the spec's piece.
            PlanInfeasible: If either anchor has no collision-free fingers-down pose.
        """
        description = move_scene_description(
            spec, self._pieces, self._board_pose, self._survey_qpos
        )
        model, index = build_scene(description)

        plan_data = make_data(model, index)
        solver = VerticalGraspSolver(model, plan_data, index)
        plan = plan_move(spec, model, plan_data, index, solver, np.random.default_rng(seed))

        data = make_data(model, index)
        executor = Executor(model, index, control_hz=self._control_hz)
        with Renderer(model, self._cameras) as renderer:
            recorder = ObservationRecorder(renderer, fps=self._fps, control_hz=self._control_hz)
            result = executor.run(data, plan, on_control_step=recorder.on_control_step)
            frames = recorder.frames(result)

        return Episode(
            spec=spec,
            instruction=plan.instruction,
            frames=frames,
            execution=result,
            engine_version=self._engine_version,
        )

    def run_all(self, specs: Sequence[MoveSpec], *, base_seed: int = 0) -> Iterator[Episode]:
        """Run a list of moves, one episode each, in order.

        Args:
            specs: The moves to run.
            base_seed: Seed of the first episode; each later one is offset by its position so
                every move solves against its own restart sequence.

        Yields:
            One episode per move, as it finishes, so a caller writing a dataset need not hold
            the whole batch in memory at once.
        """
        for offset, spec in enumerate(specs):
            yield self.run(spec, seed=base_seed + offset)


def git_engine_version(default: str = ENGINE_VERSION_UNKNOWN) -> str:
    """Read the current commit's short SHA to stamp onto generated episodes.

    Args:
        default: Value to return when the SHA cannot be read — outside a checkout, or with no
            git on the path.

    Returns:
        The short commit SHA, suffixed `-dirty` when the working tree has uncommitted changes,
        or `default` if git could not answer.
    """
    root = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if sha.returncode != 0:
            return default
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return default
    revision = sha.stdout.strip()
    if not revision:
        return default
    return f"{revision}-dirty" if dirty.stdout.strip() else revision
