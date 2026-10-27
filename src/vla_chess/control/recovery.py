"""Expert recovery from any instant of a move: where the move stands, and a plan from there.

The scripted planner lays a move out from rest. An expert that takes over a run part-way — from
wherever another controller left the arm and the piece — first has to read what the scene says
about the move's progress, then fly a plan whose first leg starts at the arm as it stands. The
plan is built from the same tilted planner the demonstrations come from, so the legs after the
first are the ones the expert would have flown anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.control.planner import (
    HOVER_ALONG_AXIS,
    MoveSpec,
    Phase,
    Plan,
    PlanInfeasible,
    Waypoint,
    plan_tilted_move,
)
from vla_chess.control.sampling import CandidateSpec, SampleKnobs, sample_candidates
from vla_chess.grasp.solver import VerticalGraspSolver
from vla_chess.sim.collision import grip_contacts
from vla_chess.sim.scene import (
    ARM_JOINT_NAMES,
    SceneIndex,
    grip_command,
    grip_radius,
    ready_command,
)

# A piece counts as standing when it rests within this of the playing surface and leans no
# further than `STANDING_TILT_DEG` off upright: the executor's own settled test.
STANDING_HEIGHT: Final = 0.01
STANDING_TILT_DEG: Final = 10.0

# How far below the hover height the TCP has to be before the hand is treated as down among the
# pieces. Above it a joint-space transit can start straight away; below it the hand is first
# lifted clear along its own axis so the transit cannot sweep the fingers through a piece.
LOW_MARGIN: Final = 0.01

# How close in the board plane the TCP must already be to a leg's target for the expert to finish
# that leg from where the hand is instead of backing off and coming in again.
DIRECT_REACH: Final = 0.02

# The arm counts as back at rest when no joint is further than this from the survey pose.
AT_REST: Final = 0.05

# Shortest leg a recovery flies, seconds: a leg that starts almost on its target still eases in
# and out rather than snapping onto it in one control step.
MIN_LEG: Final = 0.3


class NotNearTarget(PlanInfeasible):
    """A direct recovery was asked for while the hand is not near the leg it would finish."""


class Situation(Enum):
    """Where a move stands, read off the scene."""

    TO_GRASP = "to_grasp"  # the piece stands free on the board, away from its target
    IN_HAND = "in_hand"  # both jaws bear on the piece
    PLACED = "placed"  # the piece stands on its target and the hand has let go
    FINISHED = "finished"  # placed, and the arm is back at rest
    UNRECOVERABLE = "unrecoverable"  # the piece is down, airborne, or off the board


@dataclass(frozen=True)
class Assessment:
    """A move's progress at one instant, with the arm and piece poses a recovery starts from."""

    situation: Situation
    arm_qpos: npt.NDArray[np.float64]  # ARM_JOINT_NAMES order, radians
    tcp: npt.NDArray[np.float64]  # world, metres
    approach_axis: npt.NDArray[np.float64]  # world unit vector the fingers point along
    piece: npt.NDArray[np.float64]  # world position of the moved piece, metres
    piece_square: str | None


def _standing(data: mujoco.MjData, piece_body: int, surface: float) -> bool:
    """Report whether a piece rests upright on the playing surface."""
    height = float(data.xpos[piece_body][2])
    upright = float(data.xmat[piece_body].reshape(3, 3)[2, 2])
    return abs(height - surface) < STANDING_HEIGHT and upright > float(
        np.cos(np.radians(STANDING_TILT_DEG))
    )


def assess(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    spec: MoveSpec,
    solver: VerticalGraspSolver,
) -> Assessment:
    """Read how far a move has got from a simulated state.

    Args:
        model: Scene model `data` belongs to.
        data: Forward-evaluated state to read; left untouched.
        index: Name-to-id lookup for the scene.
        spec: The move being made; its source names the piece, its target the goal.
        solver: Kinematics on scratch state, used to measure the hand at the arm's configuration.

    Returns:
        The situation, and the arm, hand and piece poses a recovery plans from.
    """
    qadr = [model.jnt_qposadr[index.arm_joint_id[name]] for name in ARM_JOINT_NAMES]
    arm = np.array(data.qpos[qadr], dtype=np.float64)
    tcp = np.array(data.site_xpos[index.tcp_site_id], dtype=np.float64)
    solver.forward(arm)
    axis = np.array(solver.approach_axis(), dtype=np.float64)

    piece_body = index.piece_body_id[spec.source]
    piece = np.array(data.xpos[piece_body], dtype=np.float64)
    square = index.board.world_to_square(piece, index.board_pose)
    surface = index.board.surface_height

    if grip_contacts(model, data, piece_body).is_gripped() and square is not None:
        situation = Situation.IN_HAND
    elif square is None or not _standing(data, piece_body, surface):
        situation = Situation.UNRECOVERABLE
    elif square != spec.target:
        situation = Situation.TO_GRASP
    elif float(np.max(np.abs(arm - np.asarray(spec.survey_qpos)))) < AT_REST:
        situation = Situation.FINISHED
    else:
        situation = Situation.PLACED
    return Assessment(
        situation=situation,
        arm_qpos=arm,
        tcp=tcp,
        approach_axis=axis,
        piece=piece,
        piece_square=square,
    )


@dataclass(frozen=True)
class RecoveryCandidate:
    """One way to finish a move from an assessed instant.

    `direct` finishes the leg the hand is on from where it stands; otherwise the hand backs off
    (lifted clear first if it is down among the pieces) and the move resumes from the anchor that
    leg starts at. `sample` carries the grasp lean and placement offset, as in the funnel.
    """

    sample: CandidateSpec
    direct: bool


def recovery_candidates(
    spec: MoveSpec,
    situation: Situation,
    knobs: SampleKnobs,
    base_seed: int,
) -> list[RecoveryCandidate]:
    """Lay out the recoveries to attempt for one situation, in order of preference.

    Each sampled lean and offset is tried directly first and then with a back-off, so the run
    finishes from where the hand stands whenever that works.

    Args:
        spec: The move being made.
        situation: Where it stands.
        knobs: The lean and offset spread, as the funnel draws it.
        base_seed: Seed of the first candidate's restart search and of the spread.

    Returns:
        The candidates; empty when nothing is left to do or nothing can be done.
    """
    if situation in (Situation.FINISHED, Situation.UNRECOVERABLE):
        return []
    samples = sample_candidates(spec, knobs, base_seed, np.random.default_rng(base_seed))
    if situation is Situation.PLACED:
        return [RecoveryCandidate(samples[0], direct=False)]
    return [RecoveryCandidate(s, direct=d) for s in samples for d in (True, False)]


def _leg_duration(distance: float, nominal: float, nominal_distance: float) -> float:
    """Scale a leg's nominal duration to the distance it actually covers, seconds."""
    return float(np.clip(nominal * distance / nominal_distance, MIN_LEG, 2.0 * nominal))


def _point(values: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Turn a world vector into the tuple a waypoint carries."""
    return (float(values[0]), float(values[1]), float(values[2]))


def plan_recovery(
    candidate: RecoveryCandidate,
    assessment: Assessment,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    solver: VerticalGraspSolver,
    max_tilt_deg: float,
) -> Plan:
    """Lay out the rest of a move from an assessed instant.

    The move is re-planned with the tilted planner — the grasp at the piece's actual position, the
    grasp anchor nearest the arm — and the legs it has already done are replaced by one leg from
    the arm as it stands: straight to the current leg's target when `direct`, otherwise up clear of
    the board along the fingers when the hand is low, then on from the next anchor.

    Args:
        candidate: Which recovery to lay out.
        assessment: Where the move stands; must be one `recovery_candidates` offers plans for.
        model: Scene model to solve in.
        data: Scratch state the solving and screening poses are written into; mutated.
        index: Name-to-id lookup for the scene.
        solver: IK on `data` to search with.
        max_tilt_deg: Largest lean off vertical any anchor may hold.

    Returns:
        A plan whose first leg starts at the assessed arm configuration.

    Raises:
        NotNearTarget: If `direct` is asked for while the hand is not near the target of the leg
            it is on.
        PlanInfeasible: If the re-plan has no screened anchor.
        ValueError: If the situation leaves nothing to plan.
    """
    situation = assessment.situation
    if situation in (Situation.FINISHED, Situation.UNRECOVERABLE):
        raise ValueError(f"no recovery to plan from a {situation.value} move")
    spec = candidate.sample.move
    surface = index.board.surface_height
    arm = assessment.arm_qpos
    base, _ = plan_tilted_move(
        spec,
        model,
        data,
        index,
        solver,
        np.random.default_rng(candidate.sample.seed),
        grasp_tilt_target_deg=candidate.sample.grasp_tilt_target_deg,
        place_offset_xy=candidate.sample.place_offset_xy,
        max_tilt_deg=max_tilt_deg,
        grasp_xy=(float(assessment.piece[0]), float(assessment.piece[1])),
        grasp_prefer=(float(arm[0]), float(arm[1]), float(arm[2]), float(arm[3]), float(arm[4])),
    )
    legs = {waypoint.phase: i for i, waypoint in enumerate(base.waypoints)}
    radius = grip_radius(spec.piece_type, spec.grasp_height)
    grip = grip_command(radius, index.gripper_joint_range)
    ready = ready_command(radius, spec.standoff, index.gripper_joint_range)
    tcp = assessment.tcp
    low = float(tcp[2]) - surface < spec.hover_height - LOW_MARGIN

    if situation is Situation.TO_GRASP:
        direct_phase, resume_phase, jaw = Phase.DESCEND, Phase.APPROACH, ready
        nominal = spec.descend_duration
    elif situation is Situation.IN_HAND:
        direct_phase, resume_phase, jaw = Phase.PLACE, Phase.TRANSPORT, grip
        nominal = spec.place_duration
    else:
        direct_phase, resume_phase, jaw = Phase.RETREAT, Phase.SURVEY_END, ready
        nominal = spec.retreat_duration

    if candidate.direct:
        target = base.waypoints[legs[direct_phase]]
        assert target.position is not None
        goal = np.asarray(target.position, dtype=np.float64)
        if not low or float(np.linalg.norm(goal[:2] - tcp[:2])) > DIRECT_REACH:
            raise NotNearTarget(
                f"the hand is not near the {direct_phase.value} target to finish it directly"
            )
        first = Waypoint(
            direct_phase,
            _leg_duration(float(np.linalg.norm(goal - tcp)), nominal, HOVER_ALONG_AXIS),
            target.gripper,
            position=target.position,
            approach_axis=target.approach_axis,
        )
        rest = base.waypoints[legs[direct_phase] + 1 :]
        return _replace_waypoints(base, (first, *rest))

    lead: tuple[Waypoint, ...] = ()
    if low:
        clear = tcp + (surface + spec.hover_height - float(tcp[2])) * np.array([0.0, 0.0, 1.0])
        lift_phase = Phase.LIFT if situation is Situation.IN_HAND else Phase.RETREAT
        lead = (
            Waypoint(
                lift_phase,
                _leg_duration(float(clear[2] - tcp[2]), spec.lift_duration, HOVER_ALONG_AXIS),
                jaw,
                position=_point(clear),
                approach_axis=_point(assessment.approach_axis),
            ),
        )
    resume = base.waypoints[legs[resume_phase] :]
    if resume_phase is Phase.APPROACH:
        hover = np.asarray(base.hover_source_qpos, dtype=np.float64)
        span = float(np.max(np.abs(np.asarray(spec.survey_qpos) - hover)))
        travel = float(np.max(np.abs(arm - hover)))
        approach = resume[0]
        resume = (
            Waypoint(
                approach.phase,
                _leg_duration(travel, spec.approach_duration, max(span, 1e-6)),
                approach.gripper,
                qpos=approach.qpos,
            ),
            *resume[1:],
        )
    return _replace_waypoints(base, (*lead, *resume))


def _replace_waypoints(plan: Plan, waypoints: tuple[Waypoint, ...]) -> Plan:
    """Copy a plan with its legs swapped out."""
    return Plan(
        spec=plan.spec,
        waypoints=waypoints,
        hover_source_qpos=plan.hover_source_qpos,
        hover_target_qpos=plan.hover_target_qpos,
        instruction=plan.instruction,
    )
