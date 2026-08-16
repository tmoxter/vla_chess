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
    ARM_JOINT_NAMES,
    BIRDS_EYE_ARM_QPOS,
    GRIPPER_BODY_NAME,
    Board,
    BoardPose,
    PieceType,
    SceneIndex,
    denormalize_gripper,
    grip_command,
    grip_radius,
    ready_command,
)

ArmQpos = tuple[float, float, float, float, float]

# A fingers-down pose over the arm's own home ranks exists only within about 7 cm of the
# surface: over d7 the solver is exact at 0.070 and 13.5 mm off at 0.090, because clearing the
# board that steeply asks the wrist to fold further than it can while still pointing down. The
# hover height is set below that ceiling rather than at a clearance a taller piece would need
# — see docs/trajectory.md, which records the sweep and what it costs.
HOVER_HEIGHT: Final = 0.065

# The ends of the joint's travel. Nothing in a plan commands either: closing on a piece means
# closing to that piece (see scene.grip_command), and driving to the stop instead saturates the
# servo against whatever is in the way, while opening fully swings the jaw 100 mm past the pinch
# — through whatever stands on the next square. Approaching and leaving is `scene.ready_command`.
GRIPPER_OPEN: Final = 100.0
GRIPPER_SHUT: Final = 0.0

# How far the hand stands off the piece's axis, along the jaws' own closing direction, while it
# descends and while it leaves. It exists because only one finger moves. The fixed finger sits
# a piece-radius from the pinch and cannot get out of the way, so a descent straight down the
# piece's axis drags it past whatever the piece presents above the grasp height — on a pawn the
# head, 1 mm wider than the stem the grip is sized to. Standing off puts the piece on the
# moving jaw's side of the gap on the way in, and on the way out it is what actually releases:
# opening the jaw leaves the piece resting against the fixed finger, close enough that contact
# friction alone carried it away with the retreat until the hand backed off first.
GRASP_STANDOFF: Final = 0.005


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
    standoff: float = GRASP_STANDOFF
    survey_qpos: ArmQpos = BIRDS_EYE_ARM_QPOS
    survey_dwell: float = 1.0
    approach_duration: float = 2.5
    descend_duration: float = 1.5
    grasp_duration: float = 1.2
    lift_duration: float = 1.5
    transport_duration: float = 2.0
    place_duration: float = 1.5
    release_duration: float = 1.2
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


def _stand_off(
    point: tuple[float, float, float], axis: npt.NDArray[np.float64], distance: float
) -> tuple[float, float, float]:
    """`point` pulled back along the closing axis, putting the piece on the moving jaw's side."""
    stood = np.asarray(point, dtype=np.float64) - distance * axis
    return (float(stood[0]), float(stood[1]), float(stood[2]))


def closing_axis(
    model: mujoco.MjModel, data: mujoco.MjData, index: SceneIndex, qpos: ArmQpos
) -> npt.NDArray[np.float64]:
    """World direction the jaws open toward at `qpos`, levelled into the board's plane.

    The moving jaw sweeps out along the gripper body's own +x, so this is the direction a
    standoff has to run along. Levelling it keeps a standoff from changing the grasp height;
    at a fingers-down pose the axis is horizontal to within a degree anyway. `data` is mutated.
    """
    data.qpos[[model.jnt_qposadr[index.arm_joint_id[n]] for n in ARM_JOINT_NAMES]] = qpos
    mujoco.mj_kinematics(model, data)
    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY_NAME))
    axis = np.array(data.xmat[body].reshape(3, 3)[:, 0], dtype=np.float64)
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise PlanInfeasible(
            "the jaws close along the vertical at this pose, so no standoff "
            "direction exists in the board's plane"
        )
    return axis / norm


def solve_hover_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    solver: VerticalGraspSolver,
    square: str,
    height: float,
    rng: np.random.Generator,
    gripper_qpos: float | None = None,
) -> ArmQpos:
    """A collision-screened fingers-down configuration `height` above one square.

    Screened the way the reference grasps are, but at the opening the plan will actually be
    holding: the arm's resting self-contacts subtracted as baseline, and the piece on the
    square allowed to be touched but not penetrated — at an approach opening it should not be
    touched at all. Screening wide open instead would test a jaw that has swung 100 mm clear of
    everything the plan cares about.
    """
    baseline = baseline_self_pairs(model, data, index, np.random.default_rng(0))
    hands = hand_body_ids(model)
    piece_body = index.piece_body_id.get(square)
    opening = index.gripper_joint_range[1] if gripper_qpos is None else gripper_qpos

    def accept(qpos: npt.NDArray[np.float64]) -> bool:
        contacts = contacts_at_pose(
            model,
            data,
            index,
            qpos,
            grasped_square=square,
            baseline=baseline,
            gripper_qpos=opening,
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
    jaw_axis: Callable[..., npt.NDArray[np.float64]] = closing_axis,
) -> Plan:
    """Solve the anchor poses for one move and lay out its waypoints.

    The two survey transits are joint-space legs on purpose. Interpolating them in Cartesian
    space would ask the solver to hold the fingers down from the first control step, which
    turns the survey pose's wrist orientation into a step change rather than a motion.
    """
    board, board_pose = index.board, index.board_pose
    # A piece rests its base on the playing surface, so the height the jaws close at is the
    # same number measured from the surface or from the piece's own foot.
    radius = grip_radius(spec.piece_type, spec.grasp_height)
    grip = grip_command(radius, index.gripper_joint_range)
    ready = ready_command(radius, spec.standoff, index.gripper_joint_range)
    ready_qpos = denormalize_gripper(ready, index.gripper_joint_range)
    hover_source = solve_hover(
        model, data, index, solver, spec.source, spec.hover_height, rng, ready_qpos
    )
    hover_target = solve_hover(
        model, data, index, solver, spec.target, spec.hover_height, rng, ready_qpos
    )

    source_hover_xyz = _above(board, spec.source, board_pose, spec.hover_height)
    source_grasp_xyz = _above(board, spec.source, board_pose, spec.grasp_height)
    target_hover_xyz = _above(board, spec.target, board_pose, spec.hover_height)
    target_place_xyz = _above(board, spec.target, board_pose, spec.grasp_height)
    # The standoff runs along the jaws' own closing direction at the anchor pose, so it has to
    # be read off that pose rather than off the board: wrist roll is free, and the direction
    # the gripper opens toward is whatever the solver converged on.
    source_stand_xyz = _stand_off(
        source_grasp_xyz, jaw_axis(model, data, index, hover_source), spec.standoff
    )
    target_stand_xyz = _stand_off(
        target_place_xyz, jaw_axis(model, data, index, hover_target), spec.standoff
    )

    waypoints = (
        Waypoint(Phase.SURVEY_START, spec.survey_dwell, ready, qpos=spec.survey_qpos),
        Waypoint(Phase.APPROACH, spec.approach_duration, ready, qpos=hover_source),
        Waypoint(Phase.DESCEND, spec.descend_duration, ready, position=source_stand_xyz),
        Waypoint(Phase.GRASP, spec.grasp_duration, grip, position=source_grasp_xyz),
        Waypoint(Phase.LIFT, spec.lift_duration, grip, position=source_hover_xyz),
        Waypoint(Phase.TRANSPORT, spec.transport_duration, grip, position=target_hover_xyz),
        Waypoint(Phase.PLACE, spec.place_duration, grip, position=target_place_xyz),
        Waypoint(Phase.RELEASE, spec.release_duration, ready, position=target_stand_xyz),
        Waypoint(Phase.RETREAT, spec.retreat_duration, ready, position=target_hover_xyz),
        Waypoint(Phase.SURVEY_END, spec.return_duration, ready, qpos=spec.survey_qpos),
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
