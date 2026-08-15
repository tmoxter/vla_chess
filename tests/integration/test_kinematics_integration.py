import numpy as np
import pytest

from vla_chess.grasp import (
    MAX_APPROACH_DEG,
    MAX_TCP_ERROR,
    REFERENCE_BOARD_POSE,
    VerticalGraspSolver,
    load_reference_grasps,
)
from vla_chess.kinematics import IKSolver
from vla_chess.scene import SceneDescription, build_scene, make_data

pytestmark = [pytest.mark.slow]

REFERENCE_GRASPS = load_reference_grasps()


@pytest.fixture(scope="module")
def solvers() -> tuple[IKSolver, VerticalGraspSolver]:
    model, index = build_scene(SceneDescription(board_pose=REFERENCE_BOARD_POSE))
    data = make_data(model, index)
    return IKSolver(model, index), VerticalGraspSolver(model, data, index)


@pytest.mark.parametrize("square", sorted(REFERENCE_GRASPS))
def test_every_reference_grasp_is_recovered_without_a_seed(
    solvers: tuple[IKSolver, VerticalGraspSolver], square: str
) -> None:
    """Cold-start, which is how a planner meets a grasp pose it has no configuration for. These
    poses are also the one ground truth for what the arm can actually grip from, so a failure
    here means the wrapper does not agree with the set that was signed off by eye."""
    ik, vertical = solvers
    target = np.array(REFERENCE_GRASPS[square].qpos, dtype=np.float64)
    position, quaternion = ik.fk(target)

    result = ik.solve_from_restarts(position, quaternion, np.random.default_rng(0))

    assert result.converged, f"{square}: closest attempt was {result.position_error * 1000:.1f} mm"
    assert result.position_error <= MAX_TCP_ERROR
    _, approach_deg = vertical.measure(result.qpos, position)
    assert approach_deg <= MAX_APPROACH_DEG
