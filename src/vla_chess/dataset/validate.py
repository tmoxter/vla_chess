"""Post-hoc acceptance: turn one executed candidate into a keep or a logged rejection.

The generator does not trust a plan because it planned; it runs the candidate and asks this whether
what physically happened is worth keeping. A run is a keep only when it was never aborted, the piece
was gripped, and it ended settled and upright on its destination square — the executor's own
`placed` verdict, which already bounds the final lean to the settled cap so a slightly angled
placement still counts. Everything else is a rejection with a reason, so a move's yield can be read
back per cause.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vla_chess.control.executor import AbortReason, ExecutionResult
from vla_chess.control.planner import PlanQuality
from vla_chess.control.sampling import CandidateSpec


class RejectReason(Enum):
    PLAN_INFEASIBLE = "plan_infeasible"
    NON_FINITE_STATE = "non_finite_state"
    JOINT_LIMIT = "joint_limit"
    IK_DIVERGED = "ik_diverged"
    PIECE_DISTURBED = "piece_disturbed"
    TIMEOUT = "timeout"
    NOT_GRASPED = "not_grasped"
    WRONG_SQUARE = "wrong_square"
    NOT_SETTLED = "not_settled"


_ABORT_REASONS: dict[AbortReason, RejectReason] = {
    AbortReason.NON_FINITE_STATE: RejectReason.NON_FINITE_STATE,
    AbortReason.JOINT_LIMIT: RejectReason.JOINT_LIMIT,
    AbortReason.IK_DIVERGED: RejectReason.IK_DIVERGED,
    AbortReason.PIECE_DISTURBED: RejectReason.PIECE_DISTURBED,
    AbortReason.TIMEOUT: RejectReason.TIMEOUT,
}


@dataclass(frozen=True)
class Validation:
    """The verdict on one candidate, and enough of its story to log the yield.

    `quality` is None only when the candidate never planned (`PLAN_INFEASIBLE`): there were no
    anchors to record because no screened pose existed to attempt at all.
    """

    accepted: bool
    reason: RejectReason | None
    candidate: CandidateSpec
    quality: PlanQuality | None

    @property
    def move(self) -> str:
        """Name the move this verdict is for.

        Returns:
            The source and target squares, e.g. "d7->d5".
        """
        return f"{self.candidate.move.source}->{self.candidate.move.target}"


def infeasible(candidate: CandidateSpec) -> Validation:
    """Record a candidate that had no attemptable plan.

    Args:
        candidate: The attempt whose grasp or place square yielded no screened pose under the cap.

    Returns:
        A rejection with `PLAN_INFEASIBLE` and no quality, so the funnel can count it against yield.
    """
    return Validation(False, RejectReason.PLAN_INFEASIBLE, candidate, None)


def validate(
    candidate: CandidateSpec,
    quality: PlanQuality,
    result: ExecutionResult,
) -> Validation:
    """Judge one executed candidate against the physical outcome.

    Args:
        candidate: The attempt that was run.
        quality: The anchors the plan was built at, carried through for logging.
        result: What the executor recorded.

    Returns:
        A keep, or a rejection carrying the first cause that disqualified the run: an abort, then a
        missing grasp, then the piece off its target, then a piece that landed but did not settle
        upright.
    """
    if result.abort_reason is not None:
        return Validation(False, _ABORT_REASONS[result.abort_reason], candidate, quality)
    if not result.grasped:
        return Validation(False, RejectReason.NOT_GRASPED, candidate, quality)
    if result.final_square != candidate.move.target:
        return Validation(False, RejectReason.WRONG_SQUARE, candidate, quality)
    if not result.placed:
        return Validation(False, RejectReason.NOT_SETTLED, candidate, quality)
    return Validation(True, None, candidate, quality)
