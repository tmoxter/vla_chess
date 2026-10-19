"""The placement scorer the closed-loop rollout eval reads its success verdict from."""

from vla_chess.control.executor import Executor
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import DEFAULT_BOARD_POSE, SceneDescription, build_scene, make_data


def _rest_pawn_on(square: str) -> tuple[Executor, object]:
    """Build a scene with a lone pawn at rest on `square` and return its executor and state."""
    description = SceneDescription(
        board_pose=DEFAULT_BOARD_POSE, pieces=single_pawn_position(square)
    )
    model, index = build_scene(description)
    data = make_data(model, index)
    return Executor(model, index), data


def test_placement_scores_a_resting_pawn_on_its_target_as_placed() -> None:
    executor, data = _rest_pawn_on("d5")
    check = executor.placement(data, "d5", "d5")
    assert check.final_square == "d5"
    assert check.settled
    assert check.placed


def test_placement_on_a_pawn_resting_off_its_target_is_not_placed() -> None:
    executor, data = _rest_pawn_on("d5")
    check = executor.placement(data, "d5", "e5")
    # The pawn sits upright on d5, so it settled on a real square — just not the one asked for.
    assert check.final_square == "d5"
    assert check.settled
    assert not check.placed
