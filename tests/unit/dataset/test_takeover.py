import numpy as np
import pytest

from vla_chess.control.executor import TILTED_IK_ABORT, Executor
from vla_chess.control.planner import MoveSpec, Phase
from vla_chess.control.recovery import Situation
from vla_chess.control.sampling import SampleKnobs, plan_candidate, sample_candidates
from vla_chess.dataset.episode import move_scene_description
from vla_chess.dataset.rollouts import RecordedRollout
from vla_chess.dataset.takeover import (
    GripTrace,
    TakeoverOutcome,
    TakeoverRunner,
    carry_order,
    take_over_rollout,
    takeover_order,
)
from vla_chess.grasp.solver import VerticalGraspSolver
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    ARM_JOINT_NAMES,
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    PieceType,
    build_scene,
    make_data,
)
from vla_chess.sim.snapshot import capture_state, restore_state

MOVE = MoveSpec(
    source="d7",
    target="d6",
    piece_type=PieceType.PAWN,
    instruction_override="move the pawn one field forward",
)
PIECES = single_pawn_position(MOVE.source)


@pytest.fixture(scope="module")
def snapshots() -> dict[Phase, np.ndarray]:
    """A snapshot from the middle of each phase of a clean expert run, plus its last step."""
    model, index = build_scene(
        move_scene_description(MOVE, PIECES, DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS)
    )
    scratch = make_data(model, index)
    baseline = sample_candidates(MOVE, SampleKnobs(), 0, np.random.default_rng(0))[0]
    plan, _ = plan_candidate(
        baseline, model, scratch, index, VerticalGraspSolver(model, scratch, index)
    )
    states: list[np.ndarray] = []
    result = Executor(model, index).run(
        make_data(model, index),
        plan,
        on_control_step=lambda _, data: states.append(capture_state(model, data)),
        ik_abort=TILTED_IK_ABORT,
    )
    picked = {
        phase: states[(first + last) // 2]
        for phase, (first, last) in result.phase_boundaries.items()
    }
    picked[Phase.SURVEY_END] = states[-1]
    return picked


def test_takeover_order_leaves_out_the_leading_frames_and_covers_the_rest() -> None:
    order = takeover_order(10, 3, np.random.default_rng(0))
    assert sorted(order) == list(range(3, 10))


def test_takeover_order_of_a_rollout_shorter_than_the_skip_is_empty() -> None:
    assert takeover_order(2, 5, np.random.default_rng(0)) == []


def _trace(frames: int, held: range, release_height: float) -> GripTrace:
    gripped = np.zeros(frames, dtype=bool)
    gripped[held.start : held.stop] = True
    height = np.zeros(frames)
    height[held.stop - 1] = release_height
    return GripTrace(gripped=gripped, height=height)


def test_carry_order_tries_the_frames_before_a_mid_air_release_first() -> None:
    order = carry_order(
        _trace(40, range(10, 30), 0.02), skip=2, lead=5, per_tier=3, rng=np.random.default_rng(0)
    )
    assert set(order[:3]) <= set(range(25, 30))
    assert set(order[3:6]) <= set(range(10, 25))
    assert sorted(order) == list(range(2, 40))


def test_carry_order_puts_held_frames_first_when_the_piece_is_set_down() -> None:
    order = carry_order(
        _trace(40, range(10, 30), 0.001), skip=0, lead=5, per_tier=3, rng=np.random.default_rng(0)
    )
    assert set(order[:3]) <= set(range(10, 30))
    assert sorted(order) == list(range(40))


def test_carry_order_caps_the_held_frames_put_first() -> None:
    order = carry_order(
        _trace(400, range(10, 300), 0.02), skip=0, lead=45, per_tier=2, rng=np.random.default_rng(0)
    )
    assert set(order[:2]) <= set(range(255, 300))
    assert set(order[2:4]) <= set(range(10, 255))
    # The rest is shuffled across held and unheld frames alike.
    assert not set(order[4:20]) <= set(range(10, 300))


def test_carry_order_of_a_rollout_that_never_grips_covers_every_frame() -> None:
    trace = GripTrace(gripped=np.zeros(12, dtype=bool), height=np.zeros(12))
    order = carry_order(trace, skip=3, lead=5, per_tier=3, rng=np.random.default_rng(0))
    assert sorted(order) == list(range(3, 12))


def test_the_grip_trace_reads_the_piece_held_mid_transport(
    snapshots: dict[Phase, np.ndarray],
) -> None:
    runner = TakeoverRunner(pieces=PIECES, fps=10.0)
    trace = runner.grip_trace(
        MOVE, np.stack([snapshots[Phase.APPROACH], snapshots[Phase.TRANSPORT]])
    )
    assert trace.gripped.tolist() == [False, True]
    assert trace.height[0] < 0.003 < trace.height[1]


def test_a_takeover_mid_transport_records_the_expert_finishing_the_move(
    snapshots: dict[Phase, np.ndarray],
) -> None:
    runner = TakeoverRunner(pieces=PIECES, fps=10.0)
    outcome = runner.run(MOVE, snapshots[Phase.TRANSPORT])
    assert outcome.situation is Situation.IN_HAND
    assert outcome.episode is not None
    assert outcome.episode.execution.placed
    assert outcome.episode.instruction == "move the pawn one field forward"
    assert outcome.episode.frames[0].phase is Phase.TRANSPORT
    assert outcome.episode.frames[-1].phase is Phase.SURVEY_END


def test_the_first_frame_is_observed_from_the_snapshot(
    snapshots: dict[Phase, np.ndarray],
) -> None:
    runner = TakeoverRunner(pieces=PIECES, fps=10.0)
    episode = runner.run(MOVE, snapshots[Phase.TRANSPORT]).episode
    assert episode is not None
    model, index = build_scene(
        move_scene_description(MOVE, PIECES, DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS)
    )
    data = make_data(model, index)
    restore_state(model, data, snapshots[Phase.TRANSPORT])
    qadr = [model.jnt_qposadr[index.arm_joint_id[name]] for name in ARM_JOINT_NAMES]
    # One control step separates the snapshot from the first recorded state.
    np.testing.assert_allclose(episode.frames[0].state[:5], data.qpos[qadr], atol=0.02)


def test_a_finished_move_is_not_taken_over(snapshots: dict[Phase, np.ndarray]) -> None:
    outcome = TakeoverRunner(pieces=PIECES, fps=10.0).run(MOVE, snapshots[Phase.SURVEY_END])
    assert outcome.situation is Situation.FINISHED
    assert outcome.episode is None
    assert outcome.attempts == 0


class StubRunner:
    """Answers each frame with a scripted situation, keeping an episode where one is asked for."""

    def __init__(self, situations: dict[int, Situation], keeps: set[int]) -> None:
        self._situations = situations
        self._keeps = keeps
        self.seeds: list[int] = []

    def run(self, spec: MoveSpec, snapshot: np.ndarray, *, base_seed: int = 0) -> TakeoverOutcome:
        frame = int(snapshot[0])
        self.seeds.append(base_seed)
        episode = object() if frame in self._keeps else None
        return TakeoverOutcome(self._situations[frame], episode, 1)  # type: ignore[arg-type]


def _rollout(frames: int) -> RecordedRollout:
    return RecordedRollout(
        source="d7",
        target="d6",
        instruction="move the pawn one field forward",
        snapshots=np.arange(frames, dtype=np.float64)[:, None],
        grasped=False,
        placed=False,
        final_square="d7",
        fps=30.0,
    )


def test_a_rollout_is_taken_over_until_enough_are_kept() -> None:
    runner = StubRunner({f: Situation.TO_GRASP for f in range(20)}, keeps=set(range(20)))
    tried = take_over_rollout(
        _rollout(20), runner, wanted=3, max_attempts=10, order=list(range(20))
    )
    assert len(tried) == 3
    assert all(outcome.episode is not None for _, outcome in tried)


def test_only_recoverable_frames_count_against_the_budget() -> None:
    situations = {f: Situation.UNRECOVERABLE if f % 2 else Situation.TO_GRASP for f in range(20)}
    runner = StubRunner(situations, keeps=set())
    tried = take_over_rollout(_rollout(20), runner, wanted=3, max_attempts=4, order=list(range(20)))
    recoverable = [f for f, o in tried if o.situation is Situation.TO_GRASP]
    assert len(recoverable) == 4
    assert len(tried) > 4


def test_each_frame_is_taken_over_with_its_own_seed() -> None:
    runner = StubRunner({f: Situation.TO_GRASP for f in range(10)}, keeps=set())
    tried = take_over_rollout(
        _rollout(10),
        runner,
        wanted=1,
        max_attempts=3,
        order=[7, 4, 9],
        base_seed=1000,
    )
    assert runner.seeds == [1007, 1004, 1009]
    assert [frame for frame, _ in tried] == [7, 4, 9]
