import numpy as np
import pytest

from vla_chess.control.planner import MoveSpec
from vla_chess.control.sampling import SampleKnobs
from vla_chess.dataset.episode import EpisodeRunner
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import PieceType

pytestmark = [pytest.mark.slow, pytest.mark.render]


def _runner(source: str) -> EpisodeRunner:
    return EpisodeRunner(pieces=single_pawn_position(source), engine_version="test-sha")


def test_the_funnel_keeps_an_outer_file_push_the_vertical_planner_cannot_place() -> None:
    # a7->a6 needs a tilted grasp; the fingers-down planner had no clean hover over it, and this is
    # the whole point of the sample-then-filter path. It should place through physics and be kept.
    move = MoveSpec(
        source="a7",
        target="a6",
        piece_type=PieceType.PAWN,
        instruction_override="move the pawn one field forward",
    )
    outcome = _runner("a7").run_funnel(move, SampleKnobs(attempts=3), base_seed=0)

    assert outcome.accepted
    assert outcome.episode is not None
    assert outcome.episode.frames
    assert outcome.episode.execution.grasped
    assert outcome.episode.execution.placed
    assert outcome.episode.execution.final_square == "a6"
    # The kept episode is the last attempt logged, and it is an accept.
    assert outcome.attempts[-1].accepted


def test_the_funnel_tolerates_a_leg_that_grazes_the_old_ik_gate() -> None:
    # The kept run's IK error exceeds the fingers-down MAX_IK_ERROR (5 mm): the point of the
    # relaxed tilted abort is that such a leg still places the piece and is kept, not aborted.
    move = MoveSpec(
        source="a7",
        target="a5",
        piece_type=PieceType.PAWN,
        instruction_override="move the pawn two fields forward",
    )
    outcome = _runner("a7").run_funnel(move, SampleKnobs(attempts=3), base_seed=0)

    assert outcome.accepted
    assert outcome.episode is not None
    assert outcome.episode.execution.abort_reason is None
    assert outcome.episode.execution.diagnostics.max_ik_error > 0.005


def test_the_funnel_is_reproducible_from_its_seed() -> None:
    move = MoveSpec(
        source="d7",
        target="d5",
        piece_type=PieceType.PAWN,
        instruction_override="move the pawn two fields forward",
    )
    first = _runner("d7").run_funnel(move, SampleKnobs(attempts=2), base_seed=5)
    second = _runner("d7").run_funnel(move, SampleKnobs(attempts=2), base_seed=5)

    assert first.accepted and second.accepted
    assert first.episode is not None and second.episode is not None
    np.testing.assert_array_equal(first.episode.execution.states, second.episode.execution.states)
