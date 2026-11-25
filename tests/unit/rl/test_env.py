from collections.abc import Iterator

import numpy as np
import pytest

from vla_chess.control.planner import MoveSpec
from vla_chess.rl.env import EnvConfig, Milestones, MoveEnv, Success, end_of_chunk, milestone_bonus
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
        config = EnvConfig(max_frames=750, stop_on_success=False, stop_on_lost=False)
        assert end_of_chunk(750, True, False, config) == (1.0, False, True)
        assert end_of_chunk(750, False, True, config) == (0.0, False, True)

    def test_an_eval_episode_runs_on_past_success_and_loss(self) -> None:
        config = EnvConfig(stop_on_success=False, stop_on_lost=False)
        assert end_of_chunk(100, True, True, config) == (0.0, False, False)

    def test_nothing_happens_mid_episode(self) -> None:
        assert end_of_chunk(100, False, False, EnvConfig()) == (0.0, False, False)


class TestEndOfChunkAtDone:
    CONFIG = EnvConfig(success=Success.DONE)

    def test_done_with_the_piece_placed_ends_the_episode_rewarded(self) -> None:
        assert end_of_chunk(300, True, False, self.CONFIG, done=True, placed_at_done=True) == (
            1.0,
            True,
            False,
        )

    def test_done_with_the_piece_elsewhere_ends_the_episode_unrewarded(self) -> None:
        assert end_of_chunk(300, False, False, self.CONFIG, done=True) == (0.0, True, False)

    def test_a_placed_piece_before_the_arm_is_home_is_not_yet_a_success(self) -> None:
        assert end_of_chunk(300, True, False, self.CONFIG) == (0.0, False, False)

    def test_a_move_never_done_is_truncated_unrewarded_though_placed(self) -> None:
        assert end_of_chunk(750, True, False, self.CONFIG) == (0.0, False, True)

    def test_an_eval_episode_scores_the_done_frame_at_the_budget(self) -> None:
        config = EnvConfig(success=Success.DONE, stop_on_success=False, stop_on_lost=False)
        assert end_of_chunk(400, False, False, config, done=True, placed_at_done=True) == (
            0.0,
            False,
            False,
        )
        assert end_of_chunk(750, False, True, config, done=True, placed_at_done=True) == (
            1.0,
            False,
            True,
        )


class TestMilestoneBonus:
    CONFIG = EnvConfig(grasp_bonus=0.1, lift_bonus=0.2, place_bonus=0.3)

    def test_no_bonus_without_a_new_milestone(self) -> None:
        reached = Milestones(grasped=True)
        assert milestone_bonus(reached, reached, self.CONFIG) == 0.0

    def test_a_first_grasp_pays_its_bonus(self) -> None:
        assert milestone_bonus(Milestones(), Milestones(grasped=True), self.CONFIG) == 0.1

    def test_milestones_reached_in_one_chunk_add_up(self) -> None:
        reached = Milestones(grasped=True, lifted=True, placed=True)
        assert milestone_bonus(Milestones(), reached, self.CONFIG) == pytest.approx(0.6)

    def test_only_the_new_milestones_pay(self) -> None:
        before = Milestones(grasped=True)
        after = Milestones(grasped=True, lifted=True)
        assert milestone_bonus(before, after, self.CONFIG) == 0.2

    def test_the_default_config_pays_no_bonus(self) -> None:
        reached = Milestones(grasped=True, lifted=True, placed=True)
        assert milestone_bonus(Milestones(), reached, EnvConfig()) == 0.0


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


def test_a_grip_released_at_rest_with_the_piece_on_target_is_done_and_rewarded() -> None:
    env = MoveEnv(MOVE, EnvConfig(success=Success.DONE, max_frames=200))
    try:
        actions = _hold(env, 10)
        adr = _piece_qpos(env)
        target = env._index.board.square_to_world("d6", env._index.board_pose)
        env.data.qpos[adr : adr + 2] = np.asarray(target)[:2]
        # The arm never leaves rest; the hand is read as gripping for its first six frames.
        env._executor._holding = lambda data, body: env._frames < 6  # type: ignore[method-assign]
        results = [env.step(actions) for _ in range(3)]
        assert [r.terminated for r in results] == [False, False, True]
        assert results[-1].reward == 1.0
        outcome = results[-1].outcome
        assert outcome is not None
        assert outcome.done_frame == 20 and outcome.placed_at_done
    finally:
        env.close()


def test_an_episode_that_never_finishes_has_no_done_frame(env: MoveEnv) -> None:
    actions = _hold(env, 10)
    result = env.step(actions)
    while result.outcome is None:
        result = env.step(actions)
    assert result.outcome.done_frame is None and not result.outcome.placed_at_done


def test_a_grasp_bonus_is_paid_once_on_the_chunk_the_grip_starts() -> None:
    env = MoveEnv(MOVE, EnvConfig(max_frames=200, stop_on_lost=False, grasp_bonus=0.1))
    try:
        actions = _hold(env, 10)
        env._executor._holding = lambda data, body: 12 <= env._frames < 30  # type: ignore[method-assign]
        rewards = [env.step(actions).reward for _ in range(4)]
        assert rewards == [0.0, 0.1, 0.0, 0.0]
    finally:
        env.close()
