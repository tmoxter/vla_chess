from collections.abc import Iterator

import numpy as np
import pytest

from vla_chess.control.planner import MoveSpec
from vla_chess.rl.env import EnvConfig, MoveEnv, end_of_chunk
from vla_chess.sim.scene import PieceType

MOVE = MoveSpec(
    source="d7",
    target="d6",
    piece_type=PieceType.PAWN,
    instruction_override="move the pawn one field forward",
)


class TestEndOfChunk:
    def test_success_ends_the_episode_with_reward(self) -> None:
        assert end_of_chunk(100, True, False, EnvConfig()) == (1.0, True, False)

    def test_a_lost_piece_ends_the_episode_unrewarded(self) -> None:
        assert end_of_chunk(100, False, True, EnvConfig()) == (0.0, True, False)

    def test_the_budget_truncates_and_scores_the_last_frame(self) -> None:
        config = EnvConfig(max_frames=750, stop_on_placed=False, stop_on_lost=False)
        assert end_of_chunk(750, True, False, config) == (1.0, False, True)
        assert end_of_chunk(750, False, True, config) == (0.0, False, True)

    def test_an_eval_episode_runs_on_past_success_and_loss(self) -> None:
        config = EnvConfig(stop_on_placed=False, stop_on_lost=False)
        assert end_of_chunk(100, True, True, config) == (0.0, False, False)

    def test_nothing_happens_mid_episode(self) -> None:
        assert end_of_chunk(100, False, False, EnvConfig()) == (0.0, False, False)


@pytest.fixture
def env() -> Iterator[MoveEnv]:
    env = MoveEnv(MOVE, EnvConfig(max_frames=25))
    yield env
    env.close()


def _hold(env: MoveEnv, frames: int) -> np.ndarray:
    """Actions that command the arm to stay where it rests."""
    return np.repeat(env.reset().state[None, :], frames, axis=0)


def _piece_qpos(env: MoveEnv) -> int:
    """Address of the moved piece's free joint in qpos."""
    return int(env.model.jnt_qposadr[env.model.body_jntadr[env._piece]])


def test_reset_observes_the_wrist_camera_at_rest(env: MoveEnv) -> None:
    obs = env.reset()
    assert obs.frame.ndim == 3 and obs.frame.shape[2] == 3
    assert obs.frame.dtype == np.uint8
    assert obs.state.shape == (6,)
    assert obs.instruction == "move the pawn one field forward"


def test_a_chunk_advances_one_frame_per_action(env: MoveEnv) -> None:
    actions = _hold(env, 10)
    start = env.data.time
    result = env.step(actions)
    assert env.data.time - start == pytest.approx(10 * env.frame_substeps * env.model.opt.timestep)
    assert (result.reward, result.terminated, result.truncated) == (0.0, False, False)
    assert result.outcome is None


def test_the_frame_budget_cuts_the_last_chunk_and_truncates(env: MoveEnv) -> None:
    actions = _hold(env, 10)
    env.step(actions)
    env.step(actions)
    result = env.step(actions)
    assert result.truncated and not result.terminated
    assert result.outcome is not None
    assert result.outcome.frames == 25
    assert not result.outcome.grasped and not result.outcome.placed
    assert result.outcome.final_square == "d7"


def test_stepping_an_ended_episode_raises(env: MoveEnv) -> None:
    actions = _hold(env, 30)
    env.step(actions)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(actions)


def test_a_toppled_piece_ends_the_episode_lost(env: MoveEnv) -> None:
    actions = _hold(env, 5)
    adr = _piece_qpos(env)
    # Lay the pawn on its side: a quarter turn about the world x axis.
    env.data.qpos[adr + 3 : adr + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    result = env.step(actions)
    assert result.terminated and result.reward == 0.0
    assert result.outcome is not None and result.outcome.lost


def test_a_piece_standing_on_its_target_ends_the_episode_rewarded(env: MoveEnv) -> None:
    actions = _hold(env, 5)
    adr = _piece_qpos(env)
    target = env._index.board.square_to_world("d6", env._index.board_pose)
    env.data.qpos[adr : adr + 2] = np.asarray(target)[:2]
    result = env.step(actions)
    assert result.terminated and result.reward == 1.0
    assert result.outcome is not None
    assert result.outcome.placed and result.outcome.final_square == "d6"
    assert not result.outcome.lost


def test_reset_restores_the_scene_after_an_episode(env: MoveEnv) -> None:
    first = env.reset()
    qpos = env.data.qpos.copy()
    env.step(np.repeat(first.state[None, :], 30, axis=0))
    again = env.reset()
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(again.state, first.state)
    # GPU rasterisers may round a handful of pixels differently between identical renders.
    np.testing.assert_allclose(again.frame, first.frame, atol=1)


def test_a_seeded_reset_jitters_the_start_within_the_window() -> None:
    env = MoveEnv(MOVE, EnvConfig(piece_jitter_m=0.004))
    try:
        env.reset()
        centre = env.piece_position()
        starts = []
        for seed in range(5):
            env.reset(np.random.default_rng(seed))
            starts.append(env.piece_position())
        env.reset(np.random.default_rng(3))
        np.testing.assert_allclose(env.piece_position(), starts[3])
        offsets = np.stack(starts)[:, :2] - centre[:2]
        assert np.abs(offsets).max() <= 0.004
        assert np.abs(offsets).min(axis=0).max() > 0.0
        assert len({tuple(np.round(o, 6)) for o in offsets}) == 5
    finally:
        env.close()


def test_an_unseeded_reset_starts_on_the_centre() -> None:
    env = MoveEnv(MOVE, EnvConfig(piece_jitter_m=0.004))
    try:
        env.reset()
        centre = env.piece_position()
        env.reset(np.random.default_rng(0))
        env.reset()
        np.testing.assert_allclose(env.piece_position(), centre)
    finally:
        env.close()
