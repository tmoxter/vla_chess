"""Measurement of where the jaws' faces stand, as the engine collides them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.collision import piece_clearance
from vla_chess.scene import (
    ARM_JOINT_NAMES,
    GRIPPER_BODY_NAME,
    GRIPPER_SITE_NAME,
    MOVING_JAW_BODY_NAME,
    SceneIndex,
)

# How far either side of a square's centre the search for a face runs, and how coarsely it is
# scanned before bisection. The scan has to be finer than the thinnest thing it could step
# over: at 1 mm it cannot miss a jaw, which is 10 mm across the closing axis at its thinnest.
SEARCH_SPAN: Final = 0.060
SCAN_STEP: Final = 0.001
BISECTION_TOLERANCE: Final = 1e-5


@dataclass(frozen=True)
class JawFaces:
    """Where the two jaws' faces stand along the closing axis, measured from `gripperframe`.

    Positive is the direction the moving jaw opens toward, so `fixed` is the face a piece is
    pushed against and `moving` the face that arrives; `gap` is the free space between them.
    """

    fixed: float
    moving: float

    @property
    def gap(self) -> float:
        """Free space between the two faces."""
        return self.moving - self.fixed


def closing_axis(model: mujoco.MjModel, data: mujoco.MjData) -> npt.NDArray[np.float64]:
    """The direction the moving jaw opens toward, levelled into the board's plane.

    `data` must already carry the pose to measure at.
    """
    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY_NAME))
    axis = np.array(data.xmat[body].reshape(3, 3)[:, 0], dtype=np.float64)
    axis[2] = 0.0
    return axis / float(np.linalg.norm(axis))


def _place(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qposadr: int,
    home: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
    offset: float,
) -> None:
    """Slide the piece `offset` along the closing axis from where it started, and refresh."""
    data.qpos[joint_qposadr : joint_qposadr + 3] = home + offset * axis
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)


def _hull_hit(model: mujoco.MjModel, data: mujoco.MjData, piece_body: int, body: int) -> bool:
    """Whether the engine reports a contact between one body and the piece."""
    for c in range(data.ncon):
        contact = data.contact[c]
        pair = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if pair == {piece_body, body} and contact.dist <= 0.0:
            return True
    return False


def _bisect(hit: Callable[[float], bool], clear_at: float, hit_at: float) -> float:
    """The offset where `hit` flips, bracketed by one offset that hits and one that does not."""
    while abs(hit_at - clear_at) > BISECTION_TOLERANCE:
        middle = 0.5 * (clear_at + hit_at)
        if hit(middle):
            hit_at = middle
        else:
            clear_at = middle
    return hit_at


def measure_jaw_faces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    arm_qpos: npt.NDArray[np.float64],
    gripper_qpos: float,
    square: str,
    radius: float,
    drawn: bool = False,
) -> JawFaces:
    """Where each jaw's face stands, found by sliding one piece across the closed jaws.

    The piece is swept along the closing axis and each face is bisected for first contact, then
    the piece's own `radius` is taken back off — so the answer is the face, not the piece.
    `drawn` measures the meshes the renderer draws instead of the hulls the engine collides.
    `data` is mutated.
    """
    fixed = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY_NAME))
    moving = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOVING_JAW_BODY_NAME))
    piece_body = index.piece_body_id[square]
    joint = int(model.body_jntadr[piece_body])
    qposadr = int(model.jnt_qposadr[joint])

    data.qpos[[model.jnt_qposadr[index.arm_joint_id[n]] for n in ARM_JOINT_NAMES]] = arm_qpos
    data.qpos[model.jnt_qposadr[index.gripper_joint_id]] = gripper_qpos
    mujoco.mj_kinematics(model, data)
    axis = closing_axis(model, data)
    home = np.array(data.qpos[qposadr : qposadr + 3], dtype=np.float64)

    def hits(body: int) -> Callable[[float], bool]:
        def probe(offset: float) -> bool:
            _place(model, data, qposadr, home, axis, offset)
            if drawn:
                return piece_clearance(model, data, piece_body, frozenset({body})) <= 0.0
            return _hull_hit(model, data, piece_body, body)

        return probe

    site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIPPER_SITE_NAME))
    origin = np.array(data.site_xpos[site], dtype=np.float64)
    scan = np.arange(-SEARCH_SPAN, SEARCH_SPAN + SCAN_STEP, SCAN_STEP)
    faces: dict[str, float] = {}
    for name, body, sign in (("fixed", fixed, +1.0), ("moving", moving, -1.0)):
        probe = hits(body)
        touching = [offset for offset in scan if probe(float(offset))]
        if not touching:
            raise ValueError(
                f"the {name} jaw is not reachable by a piece of radius {radius * 1000:.1f} mm "
                f"anywhere within {SEARCH_SPAN * 1000:.0f} mm of {square}"
            )
        # The face of interest is the one the *other* jaw sees, so take the last contact on the
        # way out for the fixed finger and the first on the way in for the moving jaw.
        edge = max(touching) if sign > 0 else min(touching)
        threshold = _bisect(probe, float(edge + sign * SCAN_STEP), float(edge))
        _place(model, data, qposadr, home, axis, threshold)
        centre = float((data.qpos[qposadr : qposadr + 3] - origin) @ axis)
        faces[name] = centre - sign * radius

    _place(model, data, qposadr, home, axis, 0.0)
    return JawFaces(fixed=faces["fixed"], moving=faces["moving"])
