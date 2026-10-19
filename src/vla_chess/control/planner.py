"""Waypoint plans for a single pick-and-place move, solved against one board pose."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.grasp.solver import (
    GRASP_HEIGHT,
    MAX_APPROACH_DEG,
    MAX_TILT_DEG,
    VerticalGraspSolver,
)
from vla_chess.sim.collision import (
    baseline_self_pairs,
    contacts_at_pose,
    hand_body_ids,
    piece_clearance,
)
from vla_chess.sim.scene import (
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

# How far a tilted hover backs off its grasp or place pose, along the approach axis rather than
# straight up: a tilted pose that only exists near grasp height cannot be reached at a vertical
# hover, so the hover is a pure translation back along the fingers. Kept under the ~65 mm
# fingers-down ceiling — 60 mm put the source hover out of reach and the descent started from
# garbage. See experiment-2-plan.md, Step 2.
HOVER_ALONG_AXIS: Final = 0.040

# How far above the surface the piece's foot is aimed at release. A tilted grasp leans the gripped
# piece with the wrist, so the place aims the foot (not the TCP) at the square and lets it down to
# just above the board, where a slightly leaned piece still settles upright on the target.
RELEASE_GAP: Final = 0.003


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

    A Cartesian leg may also name an `approach_axis`: the unit direction the fingers should
    point along while it is flown, for a tilted grasp on a square no fingers-down pose reaches.
    Left unset, the leg is flown fingers-down, the default every vertical grasp uses.
    """

    phase: Phase
    duration: float
    gripper: float
    position: tuple[float, float, float] | None = None
    qpos: ArmQpos | None = None
    approach_axis: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        """Reject a leg that says nowhere to go, or takes no time getting there.

        Raises:
            ValueError: If the waypoint carries neither or both of a position and a
                configuration, if its duration is not positive, if its jaw command falls
                outside the normalised range, or if an approach axis rides a joint-space leg.
        """
        if self.approach_axis is not None and self.position is None:
            raise ValueError(
                f"{self.phase.value}: approach_axis only applies to a Cartesian leg, "
                "but this waypoint carries a qpos"
            )
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
        """Report which kind of leg this is.

        Returns:
            True when the leg is flown to a world TCP target, False when it interpolates
            joint angles.
        """
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
    # Overrides the square-naming default label. Set it when the demonstration teaches a task
    # phrased without coordinates — a lone pawn pushed forward is told "move the pawn one field
    # forward", not "from d7 to d6" — so the instruction matches what the scene actually shows.
    instruction_override: str | None = None

    def __post_init__(self) -> None:
        """Reject a move no plan can be laid out for.

        Raises:
            ValueError: If the piece is asked to move to the square it already stands on, or
                if the hover height does not clear the grasp height.
        """
        if self.source == self.target:
            raise ValueError(f"source and target are the same square: {self.source}")
        if self.hover_height <= self.grasp_height:
            raise ValueError(
                f"hover height {self.hover_height} must sit above grasp height {self.grasp_height}"
            )

    def instruction(self) -> str:
        """Phrase this move as the episode's task string.

        Returns:
            `instruction_override` verbatim when one is set; otherwise one natural-language
            sentence naming the piece, its square and its destination.
        """
        if self.instruction_override is not None:
            return self.instruction_override
        return f"move the {self.piece_type.value} from {self.source} to {self.target}"


class PlanInfeasible(RuntimeError):
    """No collision-free fingers-down pose exists for one of a plan's anchor poses."""


@dataclass(frozen=True)
class Plan:
    """A solved plan: its waypoints, its anchor configurations, and what it is trying to do."""

    spec: MoveSpec
    waypoints: tuple[Waypoint, ...]
    hover_source_qpos: ArmQpos
    hover_target_qpos: ArmQpos
    instruction: str

    def phases(self) -> tuple[Phase, ...]:
        """List the phases this plan passes through.

        Returns:
            One phase per waypoint, in the order they are flown.
        """
        return tuple(waypoint.phase for waypoint in self.waypoints)


def _above(
    board: Board, square: str, board_pose: BoardPose, height: float
) -> tuple[float, float, float]:
    """Locate the world point `height` above one square's centre."""
    point = board.square_to_world(square, board_pose)
    return (float(point[0]), float(point[1]), float(board.surface_height + height))


def _stand_off(
    point: tuple[float, float, float], axis: npt.NDArray[np.float64], distance: float
) -> tuple[float, float, float]:
    """Pull `point` back along the closing axis, onto the moving jaw's side of the gap."""
    stood = np.asarray(point, dtype=np.float64) - distance * axis
    return (float(stood[0]), float(stood[1]), float(stood[2]))


def closing_axis(
    model: mujoco.MjModel, data: mujoco.MjData, index: SceneIndex, qpos: ArmQpos
) -> npt.NDArray[np.float64]:
    """Measure the direction the jaws open toward at one configuration.

    The moving jaw sweeps out along the gripper body's own +x, so this is the direction a
    standoff has to run along. Levelling it keeps a standoff from changing the grasp height;
    at a fingers-down pose the axis is horizontal to within a degree anyway.

    Args:
        model: Scene model carrying the gripper body.
        data: State the configuration is written into and kinematics run on; mutated.
        index: Name-to-id lookup for the scene, supplying the arm joints.
        qpos: Arm configuration to measure at, in `ARM_JOINT_NAMES` order.

    Returns:
        Unit vector in the world frame, levelled into the board's plane.

    Raises:
        PlanInfeasible: If the jaws close along the vertical, leaving the axis nothing to
            project onto.
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
    prefer: ArmQpos | None = None,
) -> ArmQpos:
    """Solve a collision-screened fingers-down configuration over one square.

    Screened the way the reference grasps are, but at the opening the plan will actually be
    holding: the arm's resting self-contacts subtracted as baseline, and the piece on the
    square allowed to be touched but not penetrated — at an approach opening it should not be
    touched at all. Screening wide open instead would test a jaw that has swung 100 mm clear of
    everything the plan cares about.

    Args:
        model: Scene model to solve in.
        data: State the screening poses are written into; mutated.
        index: Name-to-id lookup for the scene, supplying the board, its pose and the pieces.
        solver: Fingers-down IK to search with.
        square: Algebraic name of the square to stand over, e.g. "e4".
        height: How far above the playing surface the TCP should sit, metres.
        rng: Source of the solver's random restart seeds.
        gripper_qpos: Jaw angle the pose is screened at, in radians. The jaw's upper limit is
            used when omitted.
        prefer: Configuration the leg into this anchor starts from, which picks the anchor out
            of the roll continuum that is nearest to it.

    Returns:
        The screened configuration, in `ARM_JOINT_NAMES` order.

    Raises:
        PlanInfeasible: If no pose over the square both reaches it within tolerance and comes
            through the screen.
    """
    baseline = baseline_self_pairs(model, data, index, np.random.default_rng(0))
    hands = hand_body_ids(model)
    piece_body = index.piece_body_id.get(square)
    opening = index.gripper_joint_range[1] if gripper_qpos is None else gripper_qpos

    def accept(qpos: npt.NDArray[np.float64]) -> bool:
        """Report whether a candidate pose touches nothing and leaves the piece clear."""
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
        square, index.board, index.board_pose, rng, height=height, accept=accept, prefer=prefer
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

    Args:
        spec: The move to plan, with the duration of each of its legs.
        model: Scene model to solve in.
        data: State the solving and screening poses are written into; mutated.
        index: Name-to-id lookup for the scene, supplying the board, its pose and the pieces.
        solver: Fingers-down IK to search with.
        rng: Source of the solver's random restart seeds.
        solve_hover: Anchor solver to call, taking the signature of `solve_hover_pose`.
        jaw_axis: Closing-direction measurement to call, taking the signature of
            `closing_axis`.

    Returns:
        The plan, carrying its waypoints and the two anchor configurations its joint-space
        legs were solved against.

    Raises:
        PlanInfeasible: If either anchor has no collision-free fingers-down pose.
    """
    board, board_pose = index.board, index.board_pose
    # A piece rests its base on the playing surface, so the height the jaws close at is the
    # same number measured from the surface or from the piece's own foot.
    radius = grip_radius(spec.piece_type, spec.grasp_height)
    grip = grip_command(radius, index.gripper_joint_range)
    ready = ready_command(radius, spec.standoff, index.gripper_joint_range)
    ready_qpos = denormalize_gripper(ready, index.gripper_joint_range)
    # Each anchor is picked nearest to where the arm arrives from: the survey pose for the
    # first, the source's own anchor for the second. Both anchors are otherwise free in roll,
    # and an unpicked one costs a transit that spends its whole duration spinning the wrist.
    hover_source = solve_hover(
        model,
        data,
        index,
        solver,
        spec.source,
        spec.hover_height,
        rng,
        ready_qpos,
        prefer=spec.survey_qpos,
    )
    hover_target = solve_hover(
        model,
        data,
        index,
        solver,
        spec.target,
        spec.hover_height,
        rng,
        ready_qpos,
        prefer=hover_source,
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


@dataclass(frozen=True)
class PlanQuality:
    """What a tilted plan's anchors cost, for the validator to weigh a run against.

    A plan is built best-effort — the lean each anchor needed is recorded rather than gated on —
    so this is how far from a clean vertical grasp the attempt started, before physics has run.
    """

    grasp_tilt_deg: float
    place_tilt_deg: float
    grasp_residual: float
    place_residual: float

    @property
    def fingers_down(self) -> bool:
        """Report whether both anchors are effectively vertical grasps.

        Returns:
            True when neither the grasp nor the place leans past the fingers-down cap, i.e. the
            sampled move happened to land on a pose the old vertical planner could also have flown.
        """
        return self.grasp_tilt_deg <= MAX_APPROACH_DEG and self.place_tilt_deg <= MAX_APPROACH_DEG


def _screen(
    model: mujoco.MjModel, data: mujoco.MjData, index: SceneIndex, square: str
) -> Callable[[npt.NDArray[np.float64]], bool]:
    """Build the collision screen a tilted anchor over one square is accepted through.

    Screens with the jaws wide open, matching how the reference grasps are solved: the near-closed
    approach opening wrongly rejects clean poses. The piece on `square` may be touched (it is the
    one being gripped); nothing else may.

    Args:
        model: Scene model to test contacts in.
        data: State the screening poses are written into; mutated.
        index: Name-to-id lookup for the scene.
        square: Square whose piece the hand is allowed to touch.

    Returns:
        A predicate that reports whether an arm configuration comes through the screen clear.
    """
    baseline = baseline_self_pairs(model, data, index, np.random.default_rng(0))
    wide_open = index.gripper_joint_range[1]

    def accept(qpos: npt.NDArray[np.float64]) -> bool:
        """Report whether a candidate pose touches nothing it should not."""
        return contacts_at_pose(
            model,
            data,
            index,
            qpos,
            grasped_square=square,
            baseline=baseline,
            gripper_qpos=wide_open,
        ).is_clear()

    return accept


def _min_tilt_anchor(
    solver: VerticalGraspSolver,
    accept: Callable[[npt.NDArray[np.float64]], bool],
    point: tuple[float, float, float],
    square: str,
    rng: np.random.Generator,
    max_tilt_deg: float,
    tilt_target_deg: float | None = None,
    prefer: ArmQpos | None = None,
) -> tuple[ArmQpos, npt.NDArray[np.float64], float, float]:
    """Solve a screened tilted anchor at a world point and read its approach axis.

    Args:
        solver: IK to search with.
        accept: Collision screen the pose must come through.
        point: World TCP target, metres.
        square: Square named in the infeasibility message.
        rng: Source of the restart seeds.
        max_tilt_deg: Largest lean off vertical the pose may hold.
        tilt_target_deg: Lean to aim for; the minimum tilt is taken when omitted.
        prefer: Configuration the arm reaches this anchor from. Passed through so the anchor
            keeps the wrist in the same roll branch and the joint-space legs do not spin it.

    Returns:
        The configuration, its world-frame approach axis (unit), its tilt off vertical in degrees,
        and its TCP error in metres.

    Raises:
        PlanInfeasible: If no pose reaches the point within tolerance, under the tilt cap, and
            through the screen.
    """
    qpos, error, tilt, ok = solver.solve_min_tilt(
        np.asarray(point, dtype=np.float64),
        rng,
        accept=accept,
        max_approach_deg=max_tilt_deg,
        tilt_target_deg=tilt_target_deg,
        prefer=prefer,
    )
    if not ok:
        raise PlanInfeasible(
            f"no clean tilted pose <= {max_tilt_deg:.0f} deg over {square}: closest attempt is "
            f"{error * 1000:.1f} mm off at {tilt:.1f} deg"
        )
    solver.forward(np.asarray(qpos, dtype=np.float64))
    axis = np.array(solver.approach_axis(), dtype=np.float64)
    axis /= float(np.linalg.norm(axis))
    return (
        (qpos[0], qpos[1], qpos[2], qpos[3], qpos[4]),
        axis,
        tilt,
        error,
    )


def plan_tilted_move(
    spec: MoveSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    solver: VerticalGraspSolver,
    rng: np.random.Generator,
    *,
    grasp_tilt_target_deg: float | None = None,
    place_offset_xy: tuple[float, float] = (0.0, 0.0),
    max_tilt_deg: float = MAX_TILT_DEG,
    hover_along_axis: float = HOVER_ALONG_AXIS,
    release_gap: float = RELEASE_GAP,
) -> tuple[Plan, PlanQuality]:
    """Lay out a tilted pick-and-place as a best-effort candidate, not a gated plan.

    Unlike `plan_move`, which forces a fingers-down pose over every anchor and raises when one
    does not exist, this takes whatever lean each square needs up to `max_tilt_deg` and flies the
    manipulation legs Cartesian with a per-leg approach axis, so the executor holds the lean rather
    than pulling back to vertical. The place aims the piece's foot at its square (offset by
    `place_offset_xy`) and sets it down just above the surface, so a leaned piece still settles on
    target. The result is meant to be executed and judged by physics, not trusted because it planned
    — a run that grazes the IK gate but places the piece is a keep.

    Args:
        spec: The move to plan, with the duration of each of its legs.
        model: Scene model to solve in.
        data: State the solving and screening poses are written into; mutated.
        index: Name-to-id lookup for the scene, supplying the board, its pose and the pieces.
        solver: IK to search with.
        rng: Source of the restart seeds.
        grasp_tilt_target_deg: Lean to aim the grasp at, degrees; the per-square minimum is taken
            when omitted. A sampler passes a value above the minimum to diversify grasp angles.
        place_offset_xy: Offset of the piece's foot from the destination square's centre, metres in
            the board plane, for sampling where inside the square it is set down.
        max_tilt_deg: Largest lean off vertical any anchor may hold.
        hover_along_axis: How far the hovers back off along the approach axis.
        release_gap: How far above the surface the foot is aimed at release.

    Returns:
        The plan and a `PlanQuality` recording the lean and residual of each anchor.

    Raises:
        PlanInfeasible: If the grasp or place square has no screened pose under the tilt cap at all
            — the move cannot be attempted, as opposed to attempted and rejected by physics.
    """
    board, board_pose = index.board, index.board_pose
    radius = grip_radius(spec.piece_type, spec.grasp_height)
    grip = grip_command(radius, index.gripper_joint_range)
    ready = ready_command(radius, spec.standoff, index.gripper_joint_range)

    grasp_xyz = _above(board, spec.source, board_pose, spec.grasp_height)
    grasp_qpos, src_axis, grasp_tilt, grasp_err = _min_tilt_anchor(
        solver,
        _screen(model, data, index, spec.source),
        grasp_xyz,
        spec.source,
        rng,
        max_tilt_deg,
        grasp_tilt_target_deg,
        prefer=spec.survey_qpos,
    )
    hover_src_xyz = _stand_off(grasp_xyz, src_axis, hover_along_axis)

    # Find the natural lean over the target, aim the foot (not the TCP) at the square, then re-solve
    # the place at that foot-compensated point: solving over the centre and commanding the shifted
    # point leaves the anchor tens of millimetres off and diverges the descent.
    target_screen = _screen(model, data, index, spec.target)
    _, place_axis0, _, _ = _min_tilt_anchor(
        solver,
        target_screen,
        _above(board, spec.target, board_pose, spec.grasp_height),
        spec.target,
        rng,
        max_tilt_deg,
    )
    target_centre = board.square_to_world(spec.target, board_pose)
    foot_target = np.array(
        [
            float(target_centre[0]) + place_offset_xy[0],
            float(target_centre[1]) + place_offset_xy[1],
            board.surface_height + release_gap,
        ]
    )
    place_xyz_arr = foot_target - spec.grasp_height * place_axis0
    place_xyz = (float(place_xyz_arr[0]), float(place_xyz_arr[1]), float(place_xyz_arr[2]))
    place_qpos, tgt_axis, place_tilt, place_err = _min_tilt_anchor(
        solver,
        target_screen,
        place_xyz,
        spec.target,
        rng,
        max_tilt_deg,
        prefer=grasp_qpos,
    )
    hover_tgt_xyz = _stand_off(place_xyz, tgt_axis, hover_along_axis)

    # Hover anchors as joint poses refined from the grasp/place pose along the axis, so they stay in
    # the same IK branch: APPROACH and SURVEY_END are flown joint-space (the survey pose is not
    # fingers-down, so a Cartesian track from it diverges) into these.
    hover_src_qpos, _, _ = solver.track(
        np.asarray(hover_src_xyz, dtype=np.float64), np.asarray(grasp_qpos), weight=0.0
    )
    hover_tgt_qpos, _, _ = solver.track(
        np.asarray(hover_tgt_xyz, dtype=np.float64), np.asarray(place_qpos), weight=0.0
    )
    hover_src_tuple: ArmQpos = tuple(float(v) for v in hover_src_qpos)  # type: ignore[assignment]
    hover_tgt_tuple: ArmQpos = tuple(float(v) for v in hover_tgt_qpos)  # type: ignore[assignment]

    src_stand = _stand_off(grasp_xyz, closing_axis(model, data, index, grasp_qpos), spec.standoff)
    tgt_stand = _stand_off(place_xyz, closing_axis(model, data, index, place_qpos), spec.standoff)

    src = (float(src_axis[0]), float(src_axis[1]), float(src_axis[2]))
    tgt = (float(tgt_axis[0]), float(tgt_axis[1]), float(tgt_axis[2]))
    waypoints = (
        Waypoint(Phase.SURVEY_START, spec.survey_dwell, ready, qpos=spec.survey_qpos),
        Waypoint(Phase.APPROACH, spec.approach_duration, ready, qpos=hover_src_tuple),
        Waypoint(
            Phase.DESCEND, spec.descend_duration, ready, position=src_stand, approach_axis=src
        ),
        Waypoint(Phase.GRASP, spec.grasp_duration, grip, position=grasp_xyz, approach_axis=src),
        Waypoint(Phase.LIFT, spec.lift_duration, grip, position=hover_src_xyz, approach_axis=src),
        Waypoint(
            Phase.TRANSPORT,
            spec.transport_duration,
            grip,
            position=hover_tgt_xyz,
            approach_axis=tgt,
        ),
        Waypoint(Phase.PLACE, spec.place_duration, grip, position=place_xyz, approach_axis=tgt),
        Waypoint(
            Phase.RELEASE, spec.release_duration, ready, position=tgt_stand, approach_axis=tgt
        ),
        Waypoint(
            Phase.RETREAT, spec.retreat_duration, ready, position=hover_tgt_xyz, approach_axis=tgt
        ),
        Waypoint(Phase.SURVEY_END, spec.return_duration, ready, qpos=spec.survey_qpos),
    )
    plan = Plan(
        spec=spec,
        waypoints=waypoints,
        hover_source_qpos=hover_src_tuple,
        hover_target_qpos=hover_tgt_tuple,
        instruction=spec.instruction(),
    )
    quality = PlanQuality(
        grasp_tilt_deg=grasp_tilt,
        place_tilt_deg=place_tilt,
        grasp_residual=grasp_err,
        place_residual=place_err,
    )
    return plan, quality


def total_duration(waypoints: tuple[Waypoint, ...]) -> float:
    """Add up how long a plan's legs take.

    Args:
        waypoints: The legs to total, in any order.

    Returns:
        Wall-clock seconds the plan is scheduled to run for.
    """
    return float(sum(waypoint.duration for waypoint in waypoints))
