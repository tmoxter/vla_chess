"""Waypoint plans for a single pick-and-place move, solved against one board pose."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.collision import (
    baseline_self_pairs,
    contacts_at_pose,
    hand_body_ids,
    piece_clearance,
)
from vla_chess.grasp import GRASP_HEIGHT, VerticalGraspSolver
from vla_chess.scene import (
    BIRDS_EYE_ARM_QPOS,
    Board,
    BoardPose,
    PieceType,
    SceneIndex,
    grip_command,
    stem_radius,
)

ArmQpos = tuple[float, float, float, float, float]

# A fingers-down pose over the arm's own home ranks exists only within about 7 cm of the
# surface: over d7 the solver is exact at 0.070 and 13.5 mm off at 0.090, because clearing the
# board that steeply asks the wrist to fold further than it can while still pointing down. The
# hover height is set below that ceiling rather than at a clearance a taller piece would need
# — see docs/trajectory.md, which records the sweep and what it costs.
HOVER_HEIGHT: Final = 0.065

GRIPPER_OPEN: Final = 100.0

# Fully shut, jaws against each other. Nothing commands it: closing on a piece means closing to
# that piece's own stem (see scene.grip_command), and driving to the stop instead saturates the
# servo against whatever is in the way. Kept as the name for the end of the joint's travel.
GRIPPER_SHUT: Final = 0.0


class Phase(Enum):
    SURVEY_START = "survey_start"
    APPROACH = "approach"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RELEASE = "release"
    RETREAT = "retreat"
    SURVEY_END = "survey_end"


@dataclass(frozen=True)
class Waypoint:
    """One leg of a plan: where to arrive, how long to take, and the jaw command on arrival.

    A waypoint is either Cartesian (`position`, a world TCP target reached fingers-down) or
    joint-space (`qpos`), never both. Joint-space legs are for the transits between the survey
    pose and the board, where no fingers-down TCP target describes the motion.
    """

    phase: Phase
    duration: float
    gripper: float
    position: tuple[float, float, float] | None = None
    qpos: ArmQpos | None = None

    def __post_init__(self) -> None:
        if (self.position is None) == (self.qpos is None):
            raise ValueError(
                f"{self.phase.value}: a waypoint takes exactly one of position or qpos, "
                f"got position={self.position}, qpos={self.qpos}"
            )
        if self.duration <= 0.0:
            raise ValueError(f"{self.phase.value}: duration must be positive, got {self.duration}")
        if not 0.0 <= self.gripper <= 100.0:
            raise ValueError(
                f"{self.phase.value}: gripper must lie in [0, 100], got {self.gripper}"
            )

    @property
    def is_cartesian(self) -> bool:
        return self.position is not None


@dataclass(frozen=True)
class MoveSpec:
    """The move to plan: which piece goes where, and how long each leg of it takes."""

    source: str
    target: str
    piece_type: PieceType
    hover_height: float = HOVER_HEIGHT
    grasp_height: float = GRASP_HEIGHT
    survey_qpos: ArmQpos = BIRDS_EYE_ARM_QPOS
    survey_dwell: float = 1.0
    approach_duration: float = 2.5
    descend_duration: float = 1.5
    grasp_duration: float = 0.8
    lift_duration: float = 1.5
    transport_duration: float = 2.0
    place_duration: float = 1.5
    release_duration: float = 0.8
    retreat_duration: float = 1.2
    return_duration: float = 2.5

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError(f"source and target are the same square: {self.source}")
        if self.hover_height <= self.grasp_height:
            raise ValueError(
                f"hover height {self.hover_height} must sit above grasp height {self.grasp_height}"
            )

    def instruction(self) -> str:
        """The episode-level task string for this move."""
        return f"move the {self.piece_type.value} from {self.source} to {self.target}"


class PlanInfeasible(RuntimeError):
    """No collision-free fingers-down pose exists for one of a plan's anchor poses."""


@dataclass(frozen=True)
class Plan:
    """A solved plan: its waypoints, the configurations its joint-space legs end at, and
    the instruction that describes it."""

    spec: MoveSpec
    waypoints: tuple[Waypoint, ...]
    hover_source_qpos: ArmQpos
    hover_target_qpos: ArmQpos
    instruction: str

    def phases(self) -> tuple[Phase, ...]:
        return tuple(waypoint.phase for waypoint in self.waypoints)


def _above(
    board: Board, square: str, board_pose: BoardPose, height: float
) -> tuple[float, float, float]:
    point = board.square_to_world(square, board_pose)
    return (float(point[0]), float(point[1]), float(board.surface_height + height))


def solve_hover_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    solver: VerticalGraspSolver,
    square: str,
    height: float,
    rng: np.random.Generator,
) -> ArmQpos:
    """A collision-screened fingers-down configuration `height` above one square.

    Screened the way the reference grasps are: jaws wide open, the arm's resting self-contacts
    subtracted as baseline, and the piece on the square allowed to be touched but not
    penetrated — with the jaws open it should not be touched at all.
    """
    baseline = baseline_self_pairs(model, data, index, np.random.default_rng(0))
    hands = hand_body_ids(model)
    piece_body = index.piece_body_id.get(square)

    def accept(qpos: npt.NDArray[np.float64]) -> bool:
        contacts = contacts_at_pose(
            model,
            data,
            index,
            qpos,
            grasped_square=square,
            baseline=baseline,
            gripper_qpos=index.gripper_joint_range[1],
        )
        if not contacts.is_clear():
            return False
        if piece_body is None:
            return True
        return piece_clearance(model, data, piece_body, hands) > 0.0

    pose = solver.solve_square(
        square, index.board, index.board_pose, rng, height=height, accept=accept
    )
    if not pose.is_within_tolerance():
        raise PlanInfeasible(
            f"no clean fingers-down pose {height * 1000:.0f} mm over {square}: "
            f"closest attempt is {pose.tcp_error * 1000:.1f} mm off at "
            f"{pose.approach_deg:.2f} deg from vertical"
        )
    return pose.qpos


def plan_move(
    spec: MoveSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    solver: VerticalGraspSolver,
    rng: np.random.Generator,
    solve_hover: Callable[..., ArmQpos] = solve_hover_pose,
) -> Plan:
    """Solve the anchor poses for one move and lay out its waypoints.

    The two survey transits are joint-space legs on purpose. Interpolating them in Cartesian
    space would ask the solver to hold the fingers down from the first control step, which
    turns the survey pose's wrist orientation into a step change rather than a motion.
    """
    board, board_pose = index.board, index.board_pose
    # A piece rests its base on the playing surface, so the height the jaws close at is the
    # same number measured from the surface or from the piece's own foot.
    grip = grip_command(stem_radius(spec.piece_type, spec.grasp_height), index.gripper_joint_range)
    hover_source = solve_hover(model, data, index, solver, spec.source, spec.hover_height, rng)
    hover_target = solve_hover(model, data, index, solver, spec.target, spec.hover_height, rng)

    source_hover_xyz = _above(board, spec.source, board_pose, spec.hover_height)
    source_grasp_xyz = _above(board, spec.source, board_pose, spec.grasp_height)
    target_hover_xyz = _above(board, spec.target, board_pose, spec.hover_height)
    target_place_xyz = _above(board, spec.target, board_pose, spec.grasp_height)

    waypoints = (
        Waypoint(Phase.SURVEY_START, spec.survey_dwell, GRIPPER_OPEN, qpos=spec.survey_qpos),
        Waypoint(Phase.APPROACH, spec.approach_duration, GRIPPER_OPEN, qpos=hover_source),
        Waypoint(Phase.DESCEND, spec.descend_duration, GRIPPER_OPEN, position=source_grasp_xyz),
        Waypoint(Phase.GRASP, spec.grasp_duration, grip, position=source_grasp_xyz),
        Waypoint(Phase.LIFT, spec.lift_duration, grip, position=source_hover_xyz),
        Waypoint(Phase.TRANSPORT, spec.transport_duration, grip, position=target_hover_xyz),
        Waypoint(Phase.PLACE, spec.place_duration, grip, position=target_place_xyz),
        Waypoint(Phase.RELEASE, spec.release_duration, GRIPPER_OPEN, position=target_place_xyz),
        Waypoint(Phase.RETREAT, spec.retreat_duration, GRIPPER_OPEN, position=target_hover_xyz),
        Waypoint(Phase.SURVEY_END, spec.return_duration, GRIPPER_OPEN, qpos=spec.survey_qpos),
    )
    return Plan(
        spec=spec,
        waypoints=waypoints,
        hover_source_qpos=hover_source,
        hover_target_qpos=hover_target,
        instruction=spec.instruction(),
    )


def total_duration(waypoints: tuple[Waypoint, ...]) -> float:
    """Wall-clock seconds a plan's legs add up to."""
    return float(sum(waypoint.duration for waypoint in waypoints))
