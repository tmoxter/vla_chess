import numpy as np
import pytest

from vla_chess.grasp import GRASP_HEIGHT
from vla_chess.planner import (
    GRIPPER_OPEN,
    GRIPPER_SHUT,
    ArmQpos,
    MoveSpec,
    Phase,
    Plan,
    Waypoint,
    plan_move,
    total_duration,
)
from vla_chess.scene import (
    DEFAULT_BOARD_POSE,
    Board,
    BoardPose,
    PieceType,
    grip_command,
    stem_radius,
)

GRIPPER_JOINT_RANGE = (-0.17453297762778586, 1.7453291995659765)

EXPECTED_PHASES = (
    Phase.SURVEY_START,
    Phase.APPROACH,
    Phase.DESCEND,
    Phase.GRASP,
    Phase.LIFT,
    Phase.TRANSPORT,
    Phase.PLACE,
    Phase.RELEASE,
    Phase.RETREAT,
    Phase.SURVEY_END,
)


class StubIndex:
    """The parts of SceneIndex the planner reads, without building a model."""

    def __init__(self, board: Board, board_pose: BoardPose) -> None:
        self.board = board
        self.board_pose = board_pose
        self.gripper_joint_range = GRIPPER_JOINT_RANGE


def stub_plan(
    source: str = "d7",
    target: str = "d6",
    piece_type: PieceType = PieceType.PAWN,
    **kwargs: float,
) -> Plan:
    index = StubIndex(Board(), DEFAULT_BOARD_POSE)
    poses = iter([(0.1, 0.2, 0.3, 0.4, 0.5), (0.6, 0.7, 0.8, 0.9, 1.0)])

    def solve_hover(*_args: object, **_kwargs: object) -> ArmQpos:
        return next(poses)

    return plan_move(
        MoveSpec(source=source, target=target, piece_type=piece_type, **kwargs),
        model=None,
        data=None,
        index=index,
        solver=None,
        rng=np.random.default_rng(0),
        solve_hover=solve_hover,
    )


def test_waypoint_needs_exactly_one_target() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Waypoint(Phase.GRASP, 1.0, GRIPPER_OPEN)
    with pytest.raises(ValueError, match="exactly one"):
        Waypoint(Phase.GRASP, 1.0, GRIPPER_OPEN, position=(0.0, 0.0, 0.0), qpos=(0.0,) * 5)


def test_waypoint_rejects_impossible_duration_and_gripper() -> None:
    with pytest.raises(ValueError, match="duration"):
        Waypoint(Phase.GRASP, 0.0, GRIPPER_OPEN, position=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="gripper"):
        Waypoint(Phase.GRASP, 1.0, 120.0, position=(0.0, 0.0, 0.0))


def test_move_spec_rejects_a_null_move() -> None:
    with pytest.raises(ValueError, match="same square"):
        MoveSpec(source="d7", target="d7", piece_type=PieceType.PAWN)


def test_move_spec_rejects_a_hover_below_the_grasp() -> None:
    with pytest.raises(ValueError, match="hover height"):
        MoveSpec(
            source="d7",
            target="d6",
            piece_type=PieceType.PAWN,
            hover_height=0.01,
            grasp_height=0.025,
        )


def test_plan_has_every_phase_in_order() -> None:
    assert stub_plan().phases() == EXPECTED_PHASES


def test_the_survey_transits_are_joint_space_and_the_rest_cartesian() -> None:
    waypoints = {w.phase: w for w in stub_plan().waypoints}
    assert waypoints[Phase.SURVEY_START].qpos is not None
    assert waypoints[Phase.APPROACH].qpos is not None
    assert waypoints[Phase.SURVEY_END].qpos is not None
    for phase in (Phase.DESCEND, Phase.GRASP, Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
        assert waypoints[phase].is_cartesian


def test_the_jaws_are_open_before_the_grasp_and_closed_until_the_release() -> None:
    waypoints = {w.phase: w for w in stub_plan().waypoints}
    expected = grip_command(stem_radius(PieceType.PAWN, GRASP_HEIGHT), GRIPPER_JOINT_RANGE)
    for phase in (Phase.SURVEY_START, Phase.APPROACH, Phase.DESCEND):
        assert waypoints[phase].gripper == GRIPPER_OPEN
    for phase in (Phase.GRASP, Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
        assert waypoints[phase].gripper == expected
    for phase in (Phase.RELEASE, Phase.RETREAT, Phase.SURVEY_END):
        assert waypoints[phase].gripper == GRIPPER_OPEN


# The grasp used to command GRIPPER_SHUT, which drives the servo into its stop against whatever
# it is holding: ~43 N of pinch on a 15 g piece. The command has to stop at the stem.
def test_the_grasp_stops_short_of_the_mechanical_stop() -> None:
    waypoints = {w.phase: w for w in stub_plan().waypoints}
    assert GRIPPER_SHUT < waypoints[Phase.GRASP].gripper < GRIPPER_OPEN


def test_a_wider_piece_is_gripped_with_the_jaws_further_apart() -> None:
    pawn = {w.phase: w for w in stub_plan(piece_type=PieceType.PAWN).waypoints}
    queen = {w.phase: w for w in stub_plan(piece_type=PieceType.QUEEN).waypoints}
    assert queen[Phase.GRASP].gripper > pawn[Phase.GRASP].gripper


def test_the_grasp_and_release_sit_over_their_own_squares() -> None:
    plan = stub_plan()
    board, pose = Board(), DEFAULT_BOARD_POSE
    waypoints = {w.phase: w for w in plan.waypoints}
    source = board.square_to_world("d7", pose)
    target = board.square_to_world("d6", pose)
    assert np.allclose(np.array(waypoints[Phase.GRASP].position)[:2], source[:2], atol=1e-9)
    assert np.allclose(np.array(waypoints[Phase.RELEASE].position)[:2], target[:2], atol=1e-9)


def test_no_waypoint_dips_below_the_playing_surface() -> None:
    surface = Board().surface_height
    for waypoint in stub_plan().waypoints:
        if waypoint.position is not None:
            assert waypoint.position[2] > surface


def test_the_grasp_is_held_at_one_height_and_the_lift_clears_it() -> None:
    waypoints = {w.phase: w for w in stub_plan().waypoints}
    assert waypoints[Phase.GRASP].position == waypoints[Phase.DESCEND].position
    assert waypoints[Phase.RELEASE].position == waypoints[Phase.PLACE].position
    assert waypoints[Phase.LIFT].position[2] > waypoints[Phase.GRASP].position[2]
    assert waypoints[Phase.RETREAT].position[2] > waypoints[Phase.PLACE].position[2]


def test_transport_stays_level() -> None:
    waypoints = {w.phase: w for w in stub_plan().waypoints}
    assert waypoints[Phase.TRANSPORT].position[2] == waypoints[Phase.LIFT].position[2]


def test_the_episode_starts_and_ends_at_the_same_survey_pose() -> None:
    waypoints = {w.phase: w for w in stub_plan().waypoints}
    assert waypoints[Phase.SURVEY_START].qpos == waypoints[Phase.SURVEY_END].qpos


def test_the_instruction_names_both_squares() -> None:
    assert stub_plan().instruction == "move the pawn from d7 to d6"


def test_total_duration_adds_the_legs_up() -> None:
    plan = stub_plan()
    assert total_duration(plan.waypoints) == pytest.approx(sum(w.duration for w in plan.waypoints))
