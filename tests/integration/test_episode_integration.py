import numpy as np
import pytest

from vla_chess.control.planner import MoveSpec
from vla_chess.dataset.episode import Episode, EpisodeRunner
from vla_chess.sim.scene import RENDER_OFFHEIGHT, RENDER_OFFWIDTH, WRIST_CAMERA_NAME, PieceType

pytestmark = [pytest.mark.slow, pytest.mark.render]


def test_runner_records_the_known_good_pawn_move() -> None:
    runner = EpisodeRunner(engine_version="test-sha")
    spec = MoveSpec(source="d7", target="d6", piece_type=PieceType.PAWN)

    episode = runner.run(spec)

    assert isinstance(episode, Episode)
    assert episode.instruction == "move the pawn from d7 to d6"
    assert episode.engine_version == "test-sha"
    assert episode.frames, "the runner captured no frames"

    first = episode.frames[0]
    assert set(first.images) == {WRIST_CAMERA_NAME}
    assert first.images[WRIST_CAMERA_NAME].shape == (RENDER_OFFHEIGHT, RENDER_OFFWIDTH, 3)
    assert first.state.shape == (6,)
    assert first.action.shape == (6,)

    # d7 -> d6 is the signed-off move: a correctly wired runner grasps it and leaves it placed.
    assert episode.succeeded
    assert episode.execution.grasped
    assert episode.execution.placed
    assert episode.execution.final_square == "d6"


def test_runner_reproduces_a_move_across_episodes() -> None:
    runner = EpisodeRunner()
    spec = MoveSpec(source="e7", target="e6", piece_type=PieceType.PAWN)

    first = runner.run(spec, seed=1)
    second = runner.run(spec, seed=1)

    np.testing.assert_array_equal(first.execution.states, second.execution.states)
    np.testing.assert_array_equal(first.execution.actions, second.execution.actions)
    assert len(first.frames) == len(second.frames)
