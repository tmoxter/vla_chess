from dataclasses import dataclass

import mujoco
import numpy as np
import pytest

from vla_chess.control.executor import TILTED_IK_ABORT, ExecutionResult, Executor
from vla_chess.control.planner import MoveSpec, Phase
from vla_chess.control.recovery import (
    Assessment,
    NotNearTarget,
    Situation,
    assess,
    plan_recovery,
    recovery_candidates,
)
from vla_chess.control.sampling import SampleKnobs, plan_candidate, sample_candidates
from vla_chess.grasp.solver import MAX_TILT_DEG, VerticalGraspSolver
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    PieceType,
    SceneDescription,
    SceneIndex,
    build_scene,
    grip_radius,
    make_data,
    pinch_offset_for,
)
from vla_chess.sim.snapshot import capture_state, restore_state

MOVE = MoveSpec(source="d7", target="d6", piece_type=PieceType.PAWN)


@dataclass(frozen=True)
class Run:
    """One clean expert run of MOVE, with a snapshot of every control step."""

    model: mujoco.MjModel
    index: SceneIndex
    executor: Executor
    result: ExecutionResult
    snapshots: list[np.ndarray]

    def state(self, step: int) -> mujoco.MjData:
        data = make_data(self.model, self.index)
        restore_state(self.model, data, self.snapshots[step])
        return data

    def assess(self, data: mujoco.MjData) -> Assessment:
        solver = VerticalGraspSolver(self.model, make_data(self.model, self.index), self.index)
        return assess(self.model, data, self.index, MOVE, solver)

    def middle_of(self, phase: Phase) -> int:
        first, last = self.result.phase_boundaries[phase]
        return (first + last) // 2

    def recover(self, data: mujoco.MjData, direct: bool) -> ExecutionResult:
        assessment = self.assess(data)
        candidate = next(
            c
            for c in recovery_candidates(MOVE, assessment.situation, SampleKnobs(), 0)
            if c.direct == direct
        )
        scratch = make_data(self.model, self.index)
        plan = plan_recovery(
            candidate,
            assessment,
            self.model,
            scratch,
            self.index,
            VerticalGraspSolver(self.model, scratch, self.index),
            MAX_TILT_DEG,
        )
        return self.executor.run(data, plan, ik_abort=TILTED_IK_ABORT)


@pytest.fixture(scope="module")
def run() -> Run:
    description = SceneDescription(
        board_pose=DEFAULT_BOARD_POSE,
        pieces=single_pawn_position(MOVE.source),
        arm_qpos=BIRDS_EYE_ARM_QPOS,
        pinch_offset=pinch_offset_for(grip_radius(MOVE.piece_type, MOVE.grasp_height)),
    )
    model, index = build_scene(description)
    scratch = make_data(model, index)
    baseline = sample_candidates(MOVE, SampleKnobs(), 0, np.random.default_rng(0))[0]
    plan, _ = plan_candidate(
        baseline, model, scratch, index, VerticalGraspSolver(model, scratch, index)
    )
    executor = Executor(model, index)
    snapshots: list[np.ndarray] = []
    result = executor.run(
        make_data(model, index),
        plan,
        on_control_step=lambda _, data: snapshots.append(capture_state(model, data)),
        ik_abort=TILTED_IK_ABORT,
    )
    assert result.succeeded
    return Run(model, index, executor, result, snapshots)


def _tip_over(run: Run, data: mujoco.MjData) -> None:
    joint = run.model.body_jntadr[run.index.piece_body_id[MOVE.source]]
    qadr = run.model.jnt_qposadr[joint]
    data.qpos[qadr + 3 : qadr + 7] = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)
    mujoco.mj_forward(run.model, data)


def _shift_piece(run: Run, data: mujoco.MjData, dx: float, dy: float) -> None:
    joint = run.model.body_jntadr[run.index.piece_body_id[MOVE.source]]
    qadr = run.model.jnt_qposadr[joint]
    data.qpos[qadr] += dx
    data.qpos[qadr + 1] += dy
    mujoco.mj_forward(run.model, data)


@pytest.mark.parametrize(
    ("phase", "situation"),
    [
        (Phase.APPROACH, Situation.TO_GRASP),
        (Phase.TRANSPORT, Situation.IN_HAND),
        (Phase.RETREAT, Situation.PLACED),
    ],
)
def test_the_situation_follows_the_move(run: Run, phase: Phase, situation: Situation) -> None:
    assert run.assess(run.state(run.middle_of(phase))).situation is situation


def test_a_move_back_at_rest_after_placing_is_finished(run: Run) -> None:
    assessment = run.assess(run.state(len(run.snapshots) - 1))
    assert assessment.situation is Situation.FINISHED
    assert recovery_candidates(MOVE, assessment.situation, SampleKnobs(), 0) == []


def test_a_piece_lying_down_is_unrecoverable(run: Run) -> None:
    data = run.state(0)
    _tip_over(run, data)
    assert run.assess(data).situation is Situation.UNRECOVERABLE


def test_a_piece_off_the_board_is_unrecoverable(run: Run) -> None:
    data = run.state(0)
    _shift_piece(run, data, 1.0, 0.0)
    assert run.assess(data).situation is Situation.UNRECOVERABLE


def test_each_lean_is_offered_directly_before_backing_off() -> None:
    candidates = recovery_candidates(MOVE, Situation.TO_GRASP, SampleKnobs(attempts=3), 0)
    assert [c.direct for c in candidates] == [True, False] * 3
    assert [c.sample.attempt for c in candidates] == [0, 0, 1, 1, 2, 2]


def test_a_placed_move_is_only_backed_away_from() -> None:
    candidates = recovery_candidates(MOVE, Situation.PLACED, SampleKnobs(), 0)
    assert [c.direct for c in candidates] == [False]


def test_finishing_directly_needs_the_hand_near_the_leg_target(run: Run) -> None:
    with pytest.raises(NotNearTarget):
        run.recover(run.state(run.middle_of(Phase.APPROACH)), direct=True)


def test_a_recovery_from_the_air_resumes_with_the_approach(run: Run) -> None:
    result = run.recover(run.state(run.middle_of(Phase.APPROACH)), direct=False)
    assert result.phases[0] is Phase.APPROACH
    assert result.succeeded


def test_a_low_hand_is_lifted_clear_straight_up_before_backing_off(run: Run) -> None:
    data = run.state(run.middle_of(Phase.DESCEND))
    start = run.assess(data).tcp
    result = run.recover(data, direct=False)
    assert result.phases[0] is Phase.RETREAT
    last = result.phases.index(Phase.APPROACH) - 1
    top = run.executor.tcp_position(result.states[last][:5])
    assert np.linalg.norm(top[:2] - start[:2]) < 0.005
    assert top[2] > start[2] + 0.005
    assert result.placed


def test_a_descent_is_finished_from_where_the_hand_is(run: Run) -> None:
    result = run.recover(run.state(run.middle_of(Phase.DESCEND)), direct=True)
    assert result.phases[0] is Phase.DESCEND
    assert result.succeeded


def test_a_piece_in_hand_is_carried_on_and_placed(run: Run) -> None:
    result = run.recover(run.state(run.middle_of(Phase.TRANSPORT)), direct=False)
    assert result.phases[0] is Phase.TRANSPORT
    assert result.placed
    assert result.abort_reason is None


def test_a_piece_pushed_off_its_square_centre_is_grasped_where_it_stands(run: Run) -> None:
    data = run.state(0)
    _shift_piece(run, data, 0.006, -0.005)
    assert run.assess(data).situation is Situation.TO_GRASP
    result = run.recover(data, direct=False)
    assert result.succeeded
