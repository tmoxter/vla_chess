import numpy as np
import pytest

from vla_chess.control.planner import MoveSpec
from vla_chess.control.sampling import SampleKnobs, sample_candidates
from vla_chess.grasp.solver import MAX_TILT_DEG
from vla_chess.sim.scene import PieceType

MOVE = MoveSpec(source="d7", target="d5", piece_type=PieceType.PAWN)


def test_knobs_reject_a_distribution_that_yields_nothing() -> None:
    with pytest.raises(ValueError, match="attempts"):
        SampleKnobs(attempts=0)
    with pytest.raises(ValueError, match="tilt_jitter_deg"):
        SampleKnobs(tilt_jitter_deg=-1.0)
    with pytest.raises(ValueError, match="place_offset_radius"):
        SampleKnobs(place_offset_radius=-0.001)


def test_the_first_candidate_is_the_clean_baseline() -> None:
    baseline = sample_candidates(MOVE, SampleKnobs(attempts=5), 7, np.random.default_rng(0))[0]
    assert baseline.attempt == 0
    assert baseline.seed == 7
    assert baseline.grasp_tilt_target_deg is None
    assert baseline.place_offset_xy == (0.0, 0.0)


def test_the_spread_has_one_candidate_per_attempt_with_distinct_seeds() -> None:
    candidates = sample_candidates(MOVE, SampleKnobs(attempts=6), 100, np.random.default_rng(1))
    assert len(candidates) == 6
    assert [c.seed for c in candidates] == [100, 101, 102, 103, 104, 105]
    assert [c.attempt for c in candidates] == list(range(6))


def test_sampled_leans_and_offsets_stay_inside_the_knobs() -> None:
    knobs = SampleKnobs(attempts=40, tilt_jitter_deg=9.0, place_offset_radius=0.005)
    for candidate in sample_candidates(MOVE, knobs, 0, np.random.default_rng(2))[1:]:
        assert candidate.grasp_tilt_target_deg is not None
        assert 0.0 <= candidate.grasp_tilt_target_deg <= 9.0
        assert np.hypot(*candidate.place_offset_xy) <= 0.005 + 1e-9


def test_a_single_attempt_is_only_the_baseline() -> None:
    candidates = sample_candidates(MOVE, SampleKnobs(attempts=1), 0, np.random.default_rng(0))
    assert len(candidates) == 1
    assert candidates[0].grasp_tilt_target_deg is None


def test_sampling_is_reproducible_from_the_rng_seed() -> None:
    first = sample_candidates(MOVE, SampleKnobs(attempts=8), 0, np.random.default_rng(42))
    second = sample_candidates(MOVE, SampleKnobs(attempts=8), 0, np.random.default_rng(42))
    assert [c.place_offset_xy for c in first] == [c.place_offset_xy for c in second]
    assert [c.grasp_tilt_target_deg for c in first] == [c.grasp_tilt_target_deg for c in second]


def test_the_default_tilt_cap_is_the_solver_cap() -> None:
    assert SampleKnobs().max_tilt_deg == MAX_TILT_DEG
