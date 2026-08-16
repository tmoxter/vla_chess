from dataclasses import dataclass
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.scene import (
    ARM_JOINT_NAMES,
    FILES,
    RANKS,
    Board,
    BoardPose,
    SceneIndex,
)

LIMIT_MARGIN: Final = 0.05
_ROLL_SAMPLES: Final = (0.0, np.pi / 2, np.pi)
_SQUARE_HIT_TOLERANCE: Final = 0.004

# Occlusion is a question about the camera, so the rays are cast against the groups the
# renderer draws — the scene's own geoms and the arm's visual meshes. Casting against every
# group instead lets the collision proxies block the view, and those are not the same solid:
# the fixed finger is collided as hulls of its own slabs (`scene.FINGER_SLAB_CUTS`), which fill
# the slot between its prongs that the drawn part sees through.
_VISIBLE_GROUPS: Final = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)


@dataclass(frozen=True)
class SurveyFraming:
    """How one arm pose frames the board through the wrist camera."""

    worst_margin: float
    worst_square: str
    out_of_frame: tuple[str, ...]
    unoccluded: int
    """Squares inside the image and not hidden by anything in front of them."""
    occluded: tuple[str, ...]
    pitch_deg: float
    height: float
    horizon_tilt_deg: float

    def frames_whole_board(self) -> bool:
        """Whether every square projects inside the image."""
        return not self.out_of_frame


def _camera_frame(
    model: mujoco.MjModel, data: mujoco.MjData, camera: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """World position and rotation of a camera, as (pos, mat) with columns right/up/back."""
    return data.cam_xpos[camera].astype(np.float64), data.cam_xmat[camera].reshape(3, 3).astype(
        np.float64
    )


def _apply(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    qpos: npt.NDArray[np.float64],
) -> None:
    """Write an arm configuration and run kinematics for it.

    Position kinematics and camera placement only: the search calls this hundreds of
    thousands of times and never reads a force.
    """
    addresses = [model.jnt_qposadr[index.arm_joint_id[name]] for name in ARM_JOINT_NAMES]
    data.qpos[addresses] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def level_rolls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    qpos: npt.NDArray[np.float64],
) -> tuple[float, ...]:
    """Every `wrist_roll` value that holds the camera horizon level.

    With the other four joints fixed the world z-component of the camera's right axis is
    `A cos(roll) + B sin(roll) + C`, so this fits those three coefficients from three
    evaluations and takes the roots in closed form. Sampling for level instead wastes
    essentially every draw.

    Both roots are returned and the caller must choose: `wrist_roll` is not the camera's
    optical axis, so the two level solutions do not merely mirror the image — one of them
    swings the camera up off the board entirely. `data` is mutated in place.
    """
    tilts = []
    candidate = np.array(qpos, dtype=float)
    for roll in _ROLL_SAMPLES:
        candidate[4] = roll
        _apply(model, data, index, candidate)
        _, mat = _camera_frame(model, data, index.wrist_camera_id)
        tilts.append(float(mat[2, 0]))
    offset = (tilts[0] + tilts[2]) / 2.0
    cosine = (tilts[0] - tilts[2]) / 2.0
    sine = tilts[1] - offset
    amplitude = float(np.hypot(cosine, sine))
    if amplitude < abs(offset) or amplitude == 0.0:
        return ()
    phase = float(np.arctan2(sine, cosine))
    spread = float(np.arccos(-offset / amplitude))
    return tuple(
        float(np.arctan2(np.sin(phase + sign * spread), np.cos(phase + sign * spread)))
        for sign in (1.0, -1.0)
    )


def framing(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    board: Board,
    board_pose: BoardPose,
    qpos: npt.NDArray[np.float64],
    rays: bool = True,
) -> SurveyFraming:
    """Measure how `qpos` frames the board, in projection and in occlusion.

    In-frame is decided by projecting square centres through the camera's own intrinsics
    rather than by reading pixels, so the margin is signed: how much of the half-frame is
    left at the worst square, negative once it has left the image. Occlusion is decided by
    ray-casting, which is what distinguishes a square the camera points at from one it can
    actually see past its own jaws. `data` is mutated in place.
    """
    _apply(model, data, index, qpos)
    camera = index.wrist_camera_id
    position, mat = _camera_frame(model, data, camera)
    half_tangent = float(np.tan(np.radians(model.cam_fovy[camera]) / 2.0))
    forward = -mat[:, 2]

    worst = np.inf
    worst_square = ""
    out_of_frame: list[str] = []
    occluded: list[str] = []
    geom_id = np.zeros(1, dtype=np.int32)
    for file in FILES:
        for rank in RANKS:
            square = f"{file}{rank}"
            target = np.asarray(board.square_to_world(square, board_pose), dtype=float)
            relative = (target - position) @ mat
            depth = -relative[2]
            if depth <= 0.0:
                out_of_frame.append(square)
                worst, worst_square = -np.inf, square
                continue
            extent = max(
                abs(relative[0] / (depth * half_tangent)),
                abs(relative[1] / (depth * half_tangent)),
            )
            margin = 1.0 - extent
            if margin < worst:
                worst, worst_square = margin, square
            if margin < 0.0:
                out_of_frame.append(square)

            if not rays or margin < 0.0:
                continue
            direction = target - position
            distance = float(np.linalg.norm(direction))
            hit = mujoco.mj_ray(
                model, data, position, direction / distance, _VISIBLE_GROUPS, 1, -1, geom_id
            )
            if geom_id[0] < 0 or hit < distance - _SQUARE_HIT_TOLERANCE:
                occluded.append(square)

    return SurveyFraming(
        worst_margin=float(worst),
        worst_square=worst_square,
        out_of_frame=tuple(out_of_frame),
        unoccluded=64 - len(out_of_frame) - len(occluded),
        occluded=tuple(occluded),
        pitch_deg=float(np.degrees(np.arcsin(-forward[2]))),
        height=float(position[2] - board.surface_height),
        horizon_tilt_deg=float(np.degrees(np.arcsin(np.clip(mat[2, 0], -1.0, 1.0)))),
    )
