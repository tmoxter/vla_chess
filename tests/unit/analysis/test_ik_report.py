import pytest

from vla_chess.analysis.ik_report import (
    MAX_MEDIAN_SOLVE_SECONDS,
    MIN_CONVERGENCE_RATE,
    IKMeasurements,
    RecoveryMeasurement,
    RoundTripMeasurement,
    ik_markdown,
)
from vla_chess.analysis.reference_ik import IKConfig


def roundtrip(converged: int = 500, median_seconds: float = 0.003) -> RoundTripMeasurement:
    return RoundTripMeasurement(
        cases=500,
        converged=converged,
        median_iterations=2.0,
        median_seconds=median_seconds,
        p95_seconds=0.004,
        max_position_error=0.0009,
    )


def recovery(from_restarts: int = 30) -> RecoveryMeasurement:
    return RecoveryMeasurement(
        total=30, from_resting=30, from_nearby=30, from_restarts=from_restarts, median_restarts=1.0
    )


def measurements(
    warm: RoundTripMeasurement | None = None, recovered: RecoveryMeasurement | None = None
) -> IKMeasurements:
    return IKMeasurements(
        warm=warm if warm is not None else roundtrip(),
        cold=roundtrip(converged=316),
        recovery=recovered if recovered is not None else recovery(),
        posture=[(1e-2, roundtrip(converged=72)), (1e-3, roundtrip())],
    )


class TestRoundTripCriteria:
    def test_a_fast_and_reliable_run_passes(self) -> None:
        assert roundtrip().meets_criteria()

    def test_a_run_below_the_convergence_floor_fails(self) -> None:
        below = int(500 * MIN_CONVERGENCE_RATE) - 1
        assert not roundtrip(converged=below).meets_criteria()

    def test_a_run_slower_than_the_budget_fails(self) -> None:
        assert not roundtrip(median_seconds=MAX_MEDIAN_SOLVE_SECONDS * 1.1).meets_criteria()


class TestRecoveryCriteria:
    def test_recovering_every_reference_pose_passes(self) -> None:
        assert recovery().meets_criteria()

    def test_missing_one_reference_pose_fails(self) -> None:
        assert not recovery(from_restarts=29).meets_criteria()


class TestMeasurementCriteria:
    def test_a_clean_sheet_passes(self) -> None:
        assert measurements().meets_criteria()

    def test_a_failing_round_trip_fails_the_whole_report(self) -> None:
        assert not measurements(warm=roundtrip(converged=100)).meets_criteria()

    def test_a_failing_recovery_fails_the_whole_report(self) -> None:
        assert not measurements(recovered=recovery(from_restarts=0)).meets_criteria()

    def test_the_cold_start_result_does_not_gate_the_report(self) -> None:
        """Cold-start convergence is a property of differential IK worth recording, not a
        threshold the solver is expected to clear — restarts are the answer to it."""
        assert measurements().meets_criteria()


@pytest.fixture(scope="module")
def document() -> str:
    return ik_markdown(measurements(), IKConfig())


class TestMarkdown:
    def test_the_report_states_how_to_regenerate_itself(self, document: str) -> None:
        assert "pixi run ik-report" in document

    def test_both_seeding_results_are_reported(self, document: str) -> None:
        assert "500/500 (100%)" in document
        assert "316/500 (63%)" in document

    def test_every_posture_cost_measured_appears(self, document: str) -> None:
        assert "| 0.01 | 72/500 (14%) |" in document
        assert "| 0.001 | 500/500 (100%) |" in document

    def test_the_configured_tolerances_are_stated(self, document: str) -> None:
        assert f"{IKConfig().position_tolerance * 1000:g} mm" in document
        assert f"{IKConfig().orientation_tolerance:g} rad" in document

    def test_the_limits_of_the_measurement_are_stated(self, document: str) -> None:
        assert "self-collision" in document
        assert "reachability.md" in document
