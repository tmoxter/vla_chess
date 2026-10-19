import numpy as np

from vla_chess.control.executor import AbortReason, ExecutionDiagnostics, ExecutionResult
from vla_chess.control.planner import MoveSpec, Phase, PlanQuality
from vla_chess.control.sampling import CandidateSpec
from vla_chess.dataset.validate import RejectReason, infeasible, validate
from vla_chess.sim.scene import PieceType

MOVE = MoveSpec(source="d7", target="d5", piece_type=PieceType.PAWN)
CANDIDATE = CandidateSpec(
    move=MOVE, seed=0, grasp_tilt_target_deg=None, place_offset_xy=(0.0, 0.0), attempt=0
)
QUALITY = PlanQuality(
    grasp_tilt_deg=8.0, place_tilt_deg=3.0, grasp_residual=0.001, place_residual=0.001
)


def _result(
    *,
    grasped: bool = True,
    placed: bool = True,
    final_square: str | None = "d5",
    abort_reason: AbortReason | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        states=np.zeros((1, 6)),
        actions=np.zeros((1, 6)),
        phases=(Phase.GRASP,),
        phase_boundaries={Phase.GRASP: (0, 0)},
        grasped=grasped,
        placed=placed,
        final_square=final_square,
        abort_reason=abort_reason,
        diagnostics=ExecutionDiagnostics(0.0, 0.0, 1, 1, 0.0),
    )


def test_a_settled_placement_on_target_is_a_keep() -> None:
    verdict = validate(CANDIDATE, QUALITY, _result())
    assert verdict.accepted
    assert verdict.reason is None
    assert verdict.move == "d7->d5"


def test_an_abort_is_rejected_with_its_mapped_reason() -> None:
    verdict = validate(
        CANDIDATE,
        QUALITY,
        _result(placed=False, final_square="d7", abort_reason=AbortReason.IK_DIVERGED),
    )
    assert not verdict.accepted
    assert verdict.reason is RejectReason.IK_DIVERGED


def test_every_abort_reason_maps_to_a_reject_reason() -> None:
    for abort in AbortReason:
        verdict = validate(CANDIDATE, QUALITY, _result(abort_reason=abort))
        assert not verdict.accepted
        assert verdict.reason is not None
        assert verdict.reason.value == abort.value


def test_a_run_that_never_gripped_is_not_grasped() -> None:
    verdict = validate(CANDIDATE, QUALITY, _result(grasped=False, placed=False, final_square="d7"))
    assert verdict.reason is RejectReason.NOT_GRASPED


def test_a_piece_left_on_the_wrong_square_is_rejected() -> None:
    verdict = validate(CANDIDATE, QUALITY, _result(placed=False, final_square="e5"))
    assert verdict.reason is RejectReason.WRONG_SQUARE


def test_a_piece_on_target_that_did_not_settle_is_rejected() -> None:
    # Right square, gripped, but tipped or airborne: placed is False while final_square matches.
    verdict = validate(CANDIDATE, QUALITY, _result(placed=False, final_square="d5"))
    assert verdict.reason is RejectReason.NOT_SETTLED


def test_the_abort_reason_is_checked_before_the_grasp() -> None:
    verdict = validate(
        CANDIDATE, QUALITY, _result(grasped=False, placed=False, abort_reason=AbortReason.TIMEOUT)
    )
    assert verdict.reason is RejectReason.TIMEOUT


def test_an_infeasible_candidate_carries_no_quality() -> None:
    verdict = infeasible(CANDIDATE)
    assert not verdict.accepted
    assert verdict.reason is RejectReason.PLAN_INFEASIBLE
    assert verdict.quality is None


def test_plan_quality_reports_fingers_down_only_when_both_anchors_are_vertical() -> None:
    assert PlanQuality(2.0, 1.0, 0.0, 0.0).fingers_down
    assert not PlanQuality(8.0, 1.0, 0.0, 0.0).fingers_down
    assert not PlanQuality(1.0, 20.0, 0.0, 0.0).fingers_down
