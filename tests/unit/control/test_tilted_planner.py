import numpy as np
import pytest

from vla_chess.control.planner import MoveSpec, Phase, Plan, plan_tilted_move
from vla_chess.grasp.solver import GRASP_HEIGHT, VerticalGraspSolver
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    PieceType,
    SceneDescription,
    SceneIndex,
    build_scene,
    make_data,
)

MOVE = MoveSpec(source="d7", target="d6", piece_type=PieceType.PAWN)


@pytest.fixture(scope="module")
def scene() -> tuple[object, SceneIndex]:
    description = SceneDescription(
        board_pose=DEFAULT_BOARD_POSE,
        pieces=single_pawn_position(MOVE.source),
        arm_qpos=BIRDS_EYE_ARM_QPOS,
    )
    return build_scene(description)


def _plan(scene: tuple[object, SceneIndex], **kwargs: object) -> Plan:
    model, index = scene
    data = make_data(model, index)
    solver = VerticalGraspSolver(model, data, index)
    plan, _ = plan_tilted_move(MOVE, model, data, index, solver, np.random.default_rng(0), **kwargs)
    return plan


def _grasp(plan: Plan) -> np.ndarray:
    (waypoint,) = [w for w in plan.waypoints if w.phase is Phase.GRASP]
    assert waypoint.position is not None
    return np.asarray(waypoint.position)


def test_the_grasp_sits_over_the_source_square_by_default(
    scene: tuple[object, SceneIndex],
) -> None:
    _, index = scene
    centre = index.board.square_to_world(MOVE.source, index.board_pose)
    grasp = _grasp(_plan(scene))
    np.testing.assert_allclose(grasp[:2], centre[:2], atol=1e-9)
    assert grasp[2] == pytest.approx(index.board.surface_height + GRASP_HEIGHT)


def test_a_grasp_point_moves_the_grasp_off_the_square_centre(
    scene: tuple[object, SceneIndex],
) -> None:
    _, index = scene
    centre = index.board.square_to_world(MOVE.source, index.board_pose)
    point = (float(centre[0]) + 0.006, float(centre[1]) - 0.004)
    grasp = _grasp(_plan(scene, grasp_xy=point))
    np.testing.assert_allclose(grasp[:2], point, atol=1e-9)
    assert grasp[2] == pytest.approx(index.board.surface_height + GRASP_HEIGHT)


def test_the_grasp_anchor_is_solved_toward_a_preferred_configuration(
    scene: tuple[object, SceneIndex],
) -> None:
    survey = _plan(scene)
    # The same reach with the wrist turned half a revolution: the anchor follows the preference
    # rather than the survey pose, so the approach does not spin the wrist back.
    turned = (*survey.hover_source_qpos[:4], survey.hover_source_qpos[4] + np.pi)
    preferred = _plan(scene, grasp_prefer=turned)
    roll_gap = abs(preferred.hover_source_qpos[4] - turned[4])
    assert roll_gap < abs(survey.hover_source_qpos[4] - turned[4])
