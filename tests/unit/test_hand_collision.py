import mujoco
import numpy as np
import pytest

from vla_chess.collision import MOVING_JAW_BODY_NAME, piece_clearance
from vla_chess.grasp import GRASP_HEIGHT, VerticalGraspSolver
from vla_chess.scene import (
    ARM_JOINT_NAMES,
    COLLISION_GROUP,
    FINGER_SLAB_CUTS,
    FIXED_FINGER_MESH,
    GRIP_SQUEEZE,
    GRIPPER_BODY_NAME,
    JAW_GAP_TABLE,
    JAW_SLAB_CUTS,
    MOVING_JAW_MESH,
    PieceColor,
    PiecePlacement,
    PieceType,
    SceneDescription,
    SceneIndex,
    build_scene,
    denormalize_gripper,
    grip_command,
    grip_radius,
    gripper_qpos_for_gap,
    jaw_gap,
    make_data,
    mesh_vertices,
    pinch_offset_for,
)

SQUARE = "d7"
Scene = tuple[mujoco.MjModel, mujoco.MjData, SceneIndex]


@pytest.fixture(scope="module")
def scene() -> Scene:
    radius = grip_radius(PieceType.PAWN, GRASP_HEIGHT)
    desc = SceneDescription(
        pieces=(PiecePlacement(SQUARE, PieceType.PAWN, PieceColor.BLACK),),
        pinch_offset=pinch_offset_for(radius),
    )
    model, index = build_scene(desc)
    return model, make_data(model, index), index


def collision_meshes(model: mujoco.MjModel, body_name: str) -> list[str]:
    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[g]))
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) == body
        and int(model.geom_group[g]) == COLLISION_GROUP
        and int(model.geom_dataid[g]) >= 0
    ]


class TestTheSplit:
    """Each jaw is collided as slabs of its own mesh, not as one hull that swallows the mouth."""

    def test_the_whole_part_is_no_longer_collided_as_one_hull(self, scene: Scene) -> None:
        model, _, _ = scene
        for body, mesh, cuts in (
            (GRIPPER_BODY_NAME, FIXED_FINGER_MESH, FINGER_SLAB_CUTS),
            (MOVING_JAW_BODY_NAME, MOVING_JAW_MESH, JAW_SLAB_CUTS),
        ):
            meshes = collision_meshes(model, body)
            assert mesh not in meshes
            assert sum(name.startswith(f"{mesh}_slab") for name in meshes) == len(cuts) + 1

    def test_the_slabs_stay_inside_the_part_they_came_from(self, scene: Scene) -> None:
        model, _, _ = scene
        for body, mesh in (
            (GRIPPER_BODY_NAME, FIXED_FINGER_MESH),
            (MOVING_JAW_BODY_NAME, MOVING_JAW_MESH),
        ):
            original = mesh_vertices(body, mesh)
            body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body))
            for geom in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
                if int(model.geom_bodyid[geom]) != body_id or "_slab" not in name:
                    continue
                slab = int(model.geom_dataid[geom])
                start = int(model.mesh_vertadr[slab])
                verts = model.mesh_vert[start : start + int(model.mesh_vertnum[slab])]
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
                placed = model.geom_pos[geom] + verts @ rotation.reshape(3, 3).T
                assert np.all(placed >= original.min(axis=0) - 1e-6), name
                assert np.all(placed <= original.max(axis=0) + 1e-6), name

    # The renderer draws groups 0-2, so a collision proxy that landed in a visible group would
    # put the hulls into the observation the policy trains on.
    def test_the_slabs_are_not_drawn(self, scene: Scene) -> None:
        model, _, _ = scene
        for body in (GRIPPER_BODY_NAME, MOVING_JAW_BODY_NAME):
            body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body))
            for geom in range(model.ngeom):
                if int(model.geom_bodyid[geom]) != body_id:
                    continue
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
                if "_slab" in name:
                    assert int(model.geom_group[geom]) == COLLISION_GROUP


class TestJawGapTable:
    """The gap between the faces, as measured rather than as fitted."""

    def test_the_gap_widens_with_the_joint(self) -> None:
        gaps = [gap for _, gap in JAW_GAP_TABLE]
        assert gaps == sorted(gaps)

    def test_an_angle_and_its_gap_are_inverses(self) -> None:
        for angle, gap in JAW_GAP_TABLE:
            assert jaw_gap(angle) == pytest.approx(gap)
            assert gripper_qpos_for_gap(gap) == pytest.approx(angle)

    # Extrapolating a piecewise curve past its last sample is a guess about a jaw feature
    # nobody slid a piece across, so both directions refuse rather than guess.
    def test_neither_direction_extrapolates(self) -> None:
        with pytest.raises(ValueError):
            jaw_gap(JAW_GAP_TABLE[-1][0] + 0.1)
        with pytest.raises(ValueError):
            gripper_qpos_for_gap(JAW_GAP_TABLE[0][1] - 0.001)


class TestTheGripTheCameraSees:
    """What the meshes do at the sized grip, which is what lands in the observation."""

    def clearances(self, scene: Scene) -> dict[str, float]:
        model, data, index = scene
        radius = grip_radius(PieceType.PAWN, GRASP_HEIGHT)
        solver = VerticalGraspSolver(model, data, index)
        pose = solver.solve_square(
            SQUARE, index.board, index.board_pose, np.random.default_rng(0), height=GRASP_HEIGHT
        )
        assert pose.is_within_tolerance()
        data.qpos[[model.jnt_qposadr[index.arm_joint_id[n]] for n in ARM_JOINT_NAMES]] = pose.qpos
        data.qpos[model.jnt_qposadr[index.gripper_joint_id]] = denormalize_gripper(
            grip_command(radius, index.gripper_joint_range), index.gripper_joint_range
        )
        mujoco.mj_kinematics(model, data)
        piece = index.piece_body_id[SQUARE]
        return {
            name: piece_clearance(
                model,
                data,
                piece,
                frozenset({int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))}),
            )
            for name in (GRIPPER_BODY_NAME, MOVING_JAW_BODY_NAME)
        }

    # The defect this guards: the engine held the piece on a hull contact while the drawn fixed
    # finger stood 10.6 mm clear of it, so the observation showed a hand gripping nothing.
    def test_both_drawn_jaws_reach_the_piece(self, scene: Scene) -> None:
        for name, clearance in self.clearances(scene).items():
            assert clearance <= 0.0, name

    def test_neither_drawn_jaw_goes_deeper_than_the_squeeze(self, scene: Scene) -> None:
        for name, clearance in self.clearances(scene).items():
            assert clearance >= -GRIP_SQUEEZE, name
