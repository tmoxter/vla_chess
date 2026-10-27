"""The executor's progress gates: a held leg re-sends its command, a clean run never waits."""

import numpy as np

from vla_chess.control.executor import (
    MAX_GATE_WAIT,
    AbortReason,
    ActionNoise,
    Executor,
    ProgressGate,
)
from vla_chess.control.planner import MoveSpec, Phase, Plan, Waypoint
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    PieceType,
    SceneDescription,
    build_scene,
    make_data,
)

OPEN_GATE = ProgressGate(lateral=1.0, vertical=1.0)
SHUT_GATE = ProgressGate(lateral=-1.0, vertical=-1.0)


def _scene() -> tuple[Executor, object, object]:
    description = SceneDescription(
        board_pose=DEFAULT_BOARD_POSE,
        pieces=single_pawn_position("d7"),
        arm_qpos=BIRDS_EYE_ARM_QPOS,
    )
    model, index = build_scene(description)
    return Executor(model, index), model, index


def _two_leg_plan() -> Plan:
    """Dwell at the survey pose with the jaws at 50, then swing the base while opening them."""
    swung = (BIRDS_EYE_ARM_QPOS[0] + 0.2, *BIRDS_EYE_ARM_QPOS[1:])
    return Plan(
        spec=MoveSpec(source="d7", target="d6", piece_type=PieceType.PAWN),
        waypoints=(
            Waypoint(Phase.SURVEY_START, 0.2, 50.0, qpos=BIRDS_EYE_ARM_QPOS),
            Waypoint(Phase.APPROACH, 0.4, 80.0, qpos=swung),
        ),
        hover_source_qpos=swung,
        hover_target_qpos=swung,
        instruction="test",
    )


def test_open_gates_leave_a_clean_run_unchanged() -> None:
    executor, model, index = _scene()
    plan = _two_leg_plan()
    clock = executor.run(make_data(model, index), plan)
    gated = executor.run(
        make_data(model, index),
        plan,
        gates={Phase.SURVEY_START: OPEN_GATE, Phase.APPROACH: OPEN_GATE},
    )
    assert np.array_equal(clock.actions, gated.actions)
    assert np.array_equal(clock.states, gated.states)


def test_a_shut_gate_holds_the_command_jaws_included_then_stalls() -> None:
    executor, model, index = _scene()
    result = executor.run(
        make_data(model, index), _two_leg_plan(), gates={Phase.APPROACH: SHUT_GATE}
    )
    assert result.abort_reason is AbortReason.STALLED
    approach = [i for i, phase in enumerate(result.phases) if phase is Phase.APPROACH]
    held = result.actions[approach]
    # Nothing past the survey dwell is ever commanded: arm and jaw stay where it left them.
    assert np.array_equal(held, np.repeat(result.actions[approach[0] - 1][None], len(held), axis=0))
    legs = round(0.4 * executor.control_hz)
    assert len(approach) == legs + round(MAX_GATE_WAIT * executor.control_hz)


def test_a_held_step_lets_the_noise_relax() -> None:
    executor, model, index = _scene()
    noise = ActionNoise(0.05, 1.5, executor.control_hz, np.random.default_rng(0))
    result = executor.run(
        make_data(model, index),
        _two_leg_plan(),
        action_noise=noise,
        gates={Phase.APPROACH: SHUT_GATE},
    )
    approach = [i for i, phase in enumerate(result.phases) if phase is Phase.APPROACH]
    # The command is held while the offset decays, so the arm converges back onto it.
    first, last = result.states[approach[0], :5], result.states[approach[-1], :5]
    command = result.actions[approach[-1], :5]
    assert np.max(np.abs(last - command)) < 0.1 * np.max(np.abs(first - command))
