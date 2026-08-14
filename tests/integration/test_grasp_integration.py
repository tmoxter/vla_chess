from dataclasses import dataclass

import mujoco
import numpy as np
import pytest

from vla_chess.cli.solve_reference_grasps import starting_position
from vla_chess.collision import baseline_self_pairs, contacts_at_pose
from vla_chess.grasp import (
    GRASP_HEIGHT,
    MAX_APPROACH_DEG,
    MAX_TCP_ERROR,
    REFERENCE_BOARD_POSE,
    VerticalGraspSolver,
    finger_axis_site_frame,
    load_reference_grasps,
)
from vla_chess.scene import SceneDescription, SceneIndex, build_scene, make_data

pytestmark = [pytest.mark.slow]

REFERENCE_GRASPS = load_reference_grasps()


@dataclass(frozen=True)
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    index: SceneIndex


@pytest.fixture(scope="module")
def scene() -> Scene:
    model, index = build_scene(
        SceneDescription(board_pose=REFERENCE_BOARD_POSE, pieces=starting_position())
    )
    return Scene(model, make_data(model, index), index)


@pytest.fixture(scope="module")
def baseline(scene: Scene) -> frozenset[tuple[int, int]]:
    return baseline_self_pairs(scene.model, scene.data, scene.index, np.random.default_rng(1))


def grasp_target(scene: Scene, square: str) -> np.ndarray:
    """World point the TCP should reach to grip the piece on `square`."""
    target = scene.index.board.square_to_world(square, REFERENCE_BOARD_POSE).astype(float)
    target[2] = scene.index.board.surface_height + GRASP_HEIGHT
    return target


class TestFingerAxis:
    def test_fingers_point_along_the_sites_x_axis_not_its_z(self, scene: Scene) -> None:
        # Guards the bug that made an earlier sweep report side approaches as top-down:
        # constraining the site's z axis constrains gripper roll, not approach direction.
        axis = finger_axis_site_frame(scene.model, scene.data, scene.index)
        assert abs(axis[0]) > 0.99
        assert abs(axis[2]) < 0.1


@pytest.mark.parametrize("square", sorted(REFERENCE_GRASPS))
class TestReferenceGraspsHold:
    def test_tcp_lands_on_the_square_at_grasp_height(self, scene: Scene, square: str) -> None:
        solver = VerticalGraspSolver(scene.model, scene.data, scene.index)
        error, angle = solver.measure(
            np.array(REFERENCE_GRASPS[square].qpos), grasp_target(scene, square)
        )
        assert error <= MAX_TCP_ERROR
        assert angle <= MAX_APPROACH_DEG

    def test_pose_is_free_of_collisions_in_the_starting_position(
        self, scene: Scene, baseline: frozenset[tuple[int, int]], square: str
    ) -> None:
        contacts = contacts_at_pose(
            scene.model,
            scene.data,
            scene.index,
            np.array(REFERENCE_GRASPS[square].qpos),
            grasped_square=square,
            baseline=baseline,
            gripper_qpos=scene.index.gripper_joint_range[1],
        )
        assert contacts.is_clear(), (
            f"{square}: world={contacts.world} self={contacts.self_collisions}"
        )


class TestSolverReproducesTheReferenceSet:
    @pytest.mark.parametrize("square", ["a8", "e7", "d6", "e5"])
    def test_resolving_lands_on_an_equivalent_pose(self, scene: Scene, square: str) -> None:
        solver = VerticalGraspSolver(scene.model, scene.data, scene.index)
        pose = solver.solve_square(
            square, scene.index.board, REFERENCE_BOARD_POSE, np.random.default_rng(0)
        )
        assert pose.is_within_tolerance()
