from itertools import pairwise

import mujoco
import numpy as np
import pytest

from vla_chess.render import CameraSpec, Renderer
from vla_chess.scene import (
    ARM_JOINT_NAMES,
    RESTING_ARM_QPOS,
    SO101_XML_PATH,
    WRIST_CAMERA_BODY,
    WRIST_CAMERA_MOUNT_RESTING_OVER_JAWS,
    WRIST_CAMERA_NAME,
    Board,
    BoardPose,
    PieceColor,
    PiecePlacement,
    PieceType,
    SceneDescription,
    build_scene,
)

pytestmark = [pytest.mark.slow]

# A joint config with the gripper hovering over the board, pointed roughly straight down —
# found by sampling forward kinematics for a pose near (0.28, 0, 0.05) with the TCP's
# approach axis close to world -z (see docs/reachability.md for the sampling method).
HOVER_OVER_BOARD_QPOS: dict[str, float] = dict(
    zip(ARM_JOINT_NAMES, [-0.03, 0.24, -0.25, 1.46, 2.21], strict=True)
)


def _set_arm_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, arm_joint_id: dict[str, int], qpos: dict[str, float]
) -> None:
    for name, value in qpos.items():
        data.qpos[model.jnt_qposadr[arm_joint_id[name]]] = value


class TestBuildSceneCompiles:
    def test_steps_1000_times_without_warnings_or_nan(self) -> None:
        desc = SceneDescription(
            pieces=(
                PiecePlacement("e4", PieceType.PAWN, PieceColor.WHITE),
                PiecePlacement("a1", PieceType.ROOK, PieceColor.WHITE),
                PiecePlacement("b1", PieceType.KNIGHT, PieceColor.WHITE),
                PiecePlacement("c1", PieceType.BISHOP, PieceColor.WHITE),
                PiecePlacement("d1", PieceType.QUEEN, PieceColor.WHITE),
                PiecePlacement("e1", PieceType.KING, PieceColor.BLACK),
            )
        )
        model, _ = build_scene(desc)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        assert data.warning.number.sum() == 0
        assert not np.isnan(data.qpos).any()
        assert not np.isnan(data.qvel).any()


class TestBaseCollisionProxy:
    def test_dropped_object_rests_on_the_base(self) -> None:
        """The vendored base body ships visual-only geometry upstream; the added convex
        proxy (scene assembly, not build_scene) must actually stop a falling object."""
        spec = mujoco.MjSpec.from_file(str(SO101_XML_PATH))
        probe = spec.worldbody.add_body(name="probe", pos=[0.017, 0.0, 0.12])
        probe.add_freejoint()
        probe.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01], mass=0.01)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe")
        settled_z = []
        for i in range(800):
            mujoco.mj_step(model, data)
            if i >= 700:
                settled_z.append(data.xpos[probe_id][2])
        z = float(np.mean(settled_z))
        assert z > 0.03, (
            f"probe object ended at z={z:.4f}, consistent with falling through the base "
            "rather than resting on it"
        )
        assert np.ptp(settled_z) < 0.01, (
            "probe object's height is still changing near the end of the simulation, "
            "consistent with falling rather than resting on the base"
        )


class TestGripperJointDirection:
    def test_fingertip_separation_increases_monotonically_with_joint_value(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(SO101_XML_PATH))
        data = mujoco.MjData(model)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        qadr = model.jnt_qposadr[joint_id]
        lo, hi = model.jnt_range[joint_id]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        static_gid, moving_gid = _jaw_geom_ids(model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        tcp = data.site_xpos[site_id].copy()
        static_tip = _closest_vertex_to(model, data, static_gid, tcp)

        separations = []
        for value in np.linspace(lo, hi, 7):
            data.qpos[:] = 0.0
            data.qpos[qadr] = value
            mujoco.mj_forward(model, data)
            moving_tip = _closest_vertex_to(model, data, moving_gid, tcp)
            separations.append(float(np.linalg.norm(moving_tip - static_tip)))

        # A small negative tolerance absorbs which-vertex-is-closest discretization noise
        # near the fully-open plateau; a real direction reversal would be orders of
        # magnitude larger than this.
        assert all(b >= a - 5e-4 for a, b in pairwise(separations)), (
            f"fingertip separation is not monotonically non-decreasing with joint value: "
            f"{separations}"
        )
        assert separations[-1] > separations[0] + 0.02, (
            "gripper joint range does not correspond to a meaningful closed-to-open sweep"
        )


def _jaw_geom_ids(model: mujoco.MjModel) -> tuple[int, int]:
    static_gid = moving_gid = -1
    for gi in range(model.ngeom):
        mesh_id = model.geom_dataid[gi]
        if mesh_id < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if name == "wrist_roll_follower_so101_v1":
            static_gid = gi
        if name == "moving_jaw_so101_v1":
            moving_gid = gi
    assert static_gid >= 0 and moving_gid >= 0
    return static_gid, moving_gid


def _closest_vertex_to(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, target: np.ndarray
) -> np.ndarray:
    """World position of a mesh geom's vertex nearest to `target`."""
    mesh_id = model.geom_dataid[geom_id]
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[start : start + count]
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)
    world = verts @ xmat.T + xpos
    return np.asarray(world[np.argmin(np.linalg.norm(world - target, axis=1))])


class TestPieceSettling:
    def test_dropped_pawn_settles_standing_on_its_square(self) -> None:
        board = Board()
        board_pose = BoardPose(0.28, 0.0, 0.0, 0.05)
        desc = SceneDescription(
            board_pose=board_pose, pieces=(PiecePlacement("d4", PieceType.PAWN, PieceColor.WHITE),)
        )
        model, index = build_scene(desc)
        data = mujoco.MjData(model)
        body_id = index.piece_body_id["d4"]
        joint_adr = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[joint_adr]

        mujoco.mj_forward(model, data)
        data.qpos[qpos_adr + 2] += 0.02  # start 2 cm above its resting placement

        for _ in range(200):
            mujoco.mj_step(model, data)

        final_xyz = data.xpos[body_id]
        target_xy = board.square_to_world("d4", board_pose)[:2]
        xy_error = float(np.linalg.norm(final_xyz[:2] - target_xy))
        assert xy_error < 0.005, f"piece settled {xy_error * 1000:.1f} mm off the square centre"

        up = data.xmat[body_id].reshape(3, 3)[:, 2]
        tilt_deg = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
        assert tilt_deg < 5.0, f"piece settled tilted {tilt_deg:.1f} deg from vertical"


class TestRestingPose:
    """RESTING_ARM_QPOS exists to satisfy one property: the jaws' swing plane is parallel to
    the board, so opening and closing reads as lateral motion from the wrist camera. These
    lock that in — the pose numbers are opaque, so a regression would otherwise be invisible
    until someone looked at a render.
    """

    def test_swing_plane_is_parallel_to_the_board(self) -> None:
        model, index = build_scene(SceneDescription())
        data = mujoco.MjData(model)
        _set_arm_qpos(
            model,
            data,
            index.arm_joint_id,
            dict(zip(ARM_JOINT_NAMES, RESTING_ARM_QPOS, strict=True)),
        )
        mujoco.mj_forward(model, data)

        gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_CAMERA_BODY)
        # The jaw hinge axis is the gripper body's local y, so the swing plane is parallel to
        # the (horizontal) board exactly when that axis is world-vertical.
        hinge_axis = data.xmat[gripper_id].reshape(3, 3)[:, 1]
        tilt_deg = float(np.degrees(np.arccos(np.clip(abs(hinge_axis[2]), -1.0, 1.0))))
        assert tilt_deg < 0.5, f"swing plane sits {tilt_deg:.2f} deg off the board plane"

    def test_jaws_separate_sideways_in_the_wrist_camera(self) -> None:
        desc = SceneDescription(wrist_camera_mount=WRIST_CAMERA_MOUNT_RESTING_OVER_JAWS)
        model, index = build_scene(desc)
        data = mujoco.MjData(model)
        _set_arm_qpos(
            model,
            data,
            index.arm_joint_id,
            dict(zip(ARM_JOINT_NAMES, RESTING_ARM_QPOS, strict=True)),
        )
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA_NAME)
        _, moving_gid = _jaw_geom_ids(model)
        lo, hi = model.jnt_range[index.gripper_joint_id]

        image_xy = []
        for value in (lo, hi):
            data.qpos[model.jnt_qposadr[index.gripper_joint_id]] = value
            mujoco.mj_forward(model, data)
            # Camera frame: x right, y up, looking down -z, so -z is the depth in front.
            offset = data.cam_xmat[camera_id].reshape(3, 3).T @ (
                data.geom_xpos[moving_gid] - data.cam_xpos[camera_id]
            )
            assert offset[2] < 0, "moving jaw is behind the wrist camera at the resting pose"
            image_xy.append(offset[:2] / -offset[2])

        travel = image_xy[1] - image_xy[0]
        assert abs(travel[0]) > 0.3, (
            f"jaw barely moves sideways in frame between closed and open: du={travel[0]:+.3f}"
        )
        # Predominantly sideways, not merely non-zero. The floor is not 0 deg: the swing
        # direction is tipped along the finger/view axis, so part of it always projects as
        # approach-in-depth (see WRIST_CAMERA_MOUNT_RESTING_OVER_JAWS).
        assert abs(travel[0]) > abs(travel[1]), (
            f"jaw travel reads more vertical than horizontal: du={travel[0]:+.3f}, "
            f"dv={travel[1]:+.3f}"
        )


@pytest.mark.render
class TestSceneRendering:
    def test_wrist_camera_sees_a_nonuniform_scene(self) -> None:
        desc = SceneDescription(pieces=(PiecePlacement("e4", PieceType.PAWN, PieceColor.WHITE),))
        model, index = build_scene(desc)
        data = mujoco.MjData(model)
        _set_arm_qpos(model, data, index.arm_joint_id, HOVER_OVER_BOARD_QPOS)
        mujoco.mj_forward(model, data)
        with Renderer(model, [CameraSpec(name="wrist", width=128, height=128)]) as renderer:
            frame = renderer.render(data)["wrist"]
        assert frame.std() > 10.0
