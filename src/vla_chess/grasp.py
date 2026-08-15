"""Vertical (fingers-down) grasp poses: the IK that finds them, and the solved reference set."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.scene import (
    ARM_JOINT_NAMES,
    DEFAULT_BOARD_POSE,
    GRIPPER_BODY_NAME,
    SURFACE_HEIGHT,
    Board,
    BoardPose,
    SceneIndex,
)

REFERENCE_GRASPS_PATH: Final = Path(__file__).resolve().parent / "assets" / "reference_grasps.json"

# These grasps are solved at the scene's own board pose, not at a placement of their own. An
# earlier revision solved them 2 cm further out, on the strength of a fingers-down sweep that
# scored that distance two squares better; the cost was that the reference set silently did not
# apply to a default scene, landing 20 mm off every square centre — under a piece's foot radius
# of correct, and well past the 3 mm this module accepts. The scene pose is the one that was
# confirmed by eye, so it wins, and the sweep now lives in docs/grasp-poses.md as history.
REFERENCE_BOARD_POSE: Final = DEFAULT_BOARD_POSE

# How far above the playing surface the TCP sits when closing on a piece: on the stem of every
# piece type, above the foot and below the head.
GRASP_HEIGHT: Final = 0.025
GRASP_Z: Final = SURFACE_HEIGHT + GRASP_HEIGHT

WORLD_DOWN: Final = (0.0, 0.0, -1.0)

# Acceptance thresholds for a solved pose, also the tolerances the reference set is checked
# against: a TCP within a fifth of a piece's foot radius of the square centre, and an approach
# the eye reads as vertical.
MAX_TCP_ERROR: Final = 0.003
MAX_APPROACH_DEG: Final = 3.0

_IK_ITERATIONS: Final = 120
_IK_DAMPING: Final = 1e-3
_IK_MAX_STEP: Final = 0.3
_IK_RESTARTS: Final = 6
# Restart budget when the caller also screens poses (for collisions, say). Far larger than
# `_IK_RESTARTS` because reaching the target is the easy half: the fingers-down constraint
# leaves wrist_roll free, so a square typically has a continuum of in-tolerance solutions of
# which only some clear the arm's own body, and the first one converged on is not usually one
# of them. Squares with no clean solution at all spend the whole budget proving it, which is
# cheap enough (a few seconds each) to be worth paying for a correct verdict.
_IK_SCREENED_RESTARTS: Final = 60
_FD_EPS: Final = 1e-6


@dataclass(frozen=True)
class GraspPose:
    """One solved arm configuration that grips a square from straight above.

    `qpos` is in `ARM_JOINT_NAMES` order and excludes the gripper joint, which is free: the
    same configuration serves the open-jaw hover and the closed-jaw hold.
    """

    square: str
    qpos: tuple[float, float, float, float, float]
    tcp_error: float
    approach_deg: float

    def is_within_tolerance(self) -> bool:
        """Whether this pose meets the module's TCP and approach-angle thresholds."""
        return self.tcp_error <= MAX_TCP_ERROR and self.approach_deg <= MAX_APPROACH_DEG


def finger_axis_site_frame(
    model: mujoco.MjModel, data: mujoco.MjData, index: SceneIndex
) -> npt.NDArray[np.float64]:
    """Unit vector, in the TCP site's own frame, along which the fingers stick out.

    Worth measuring rather than assuming: it is very nearly the site's local x, so code that
    reasons about approach direction via the site's z axis constrains the gripper's roll
    instead of its approach, and selects side approaches while reporting them as top-down.
    """
    mujoco.mj_kinematics(model, data)
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY_NAME))
    world = data.site_xpos[index.tcp_site_id] - data.xpos[body_id]
    world = world / np.linalg.norm(world)
    rotation = data.site_xmat[index.tcp_site_id].reshape(3, 3)
    return np.asarray(rotation.T @ world, dtype=np.float64)


def _arm_addresses(model: mujoco.MjModel, index: SceneIndex) -> npt.NDArray[np.int64]:
    return np.array(
        [model.jnt_qposadr[index.arm_joint_id[name]] for name in ARM_JOINT_NAMES], dtype=np.int64
    )


def joint_limits(
    model: mujoco.MjModel, index: SceneIndex
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Lower and upper bounds of the five arm joints, in `ARM_JOINT_NAMES` order."""
    ids = [index.arm_joint_id[name] for name in ARM_JOINT_NAMES]
    lo = np.array([model.jnt_range[j][0] for j in ids], dtype=np.float64)
    hi = np.array([model.jnt_range[j][1] for j in ids], dtype=np.float64)
    return lo, hi


class VerticalGraspSolver:
    """Damped least-squares IK for placing the TCP at a world point with the fingers down.

    Five joints against six residuals (three position, three approach-axis) — the axis
    residual is rank two, so the system is square in practice and wrist_roll stays free.
    The Jacobian is taken by finite differences: the residual mixes a site position with a
    rotated body axis, and the analytic form of the latter is easy to get sign-wrong.

    Mutates the `data` it is constructed with.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, index: SceneIndex) -> None:
        self._model = model
        self._data = data
        self._index = index
        self._qadr = _arm_addresses(model, index)
        self._lo, self._hi = joint_limits(model, index)
        self._finger_axis = finger_axis_site_frame(model, data, index)
        self._down = np.array(WORLD_DOWN, dtype=np.float64)

    @property
    def finger_axis(self) -> npt.NDArray[np.float64]:
        """The finger axis in the TCP site's frame, as measured from this model."""
        return self._finger_axis

    def forward(self, qpos: npt.NDArray[np.float64]) -> None:
        """Write an arm configuration into `data` and run forward kinematics."""
        self._data.qpos[self._qadr] = qpos
        mujoco.mj_kinematics(self._model, self._data)

    def approach_axis(self) -> npt.NDArray[np.float64]:
        """The current world-frame direction the fingers point in."""
        rotation = self._data.site_xmat[self._index.tcp_site_id].reshape(3, 3)
        return np.asarray(rotation @ self._finger_axis, dtype=np.float64)

    def _residual(
        self, qpos: npt.NDArray[np.float64], target: npt.NDArray[np.float64], weight: float
    ) -> npt.NDArray[np.float64]:
        self.forward(qpos)
        position = self._data.site_xpos[self._index.tcp_site_id] - target
        return np.concatenate([position, weight * (self.approach_axis() - self._down)])

    def _refine(
        self, seed: npt.NDArray[np.float64], target: npt.NDArray[np.float64], weight: float
    ) -> npt.NDArray[np.float64]:
        qpos = np.clip(seed, self._lo, self._hi)
        previous = np.inf
        for _ in range(_IK_ITERATIONS):
            residual = self._residual(qpos, target, weight)
            jacobian = np.empty((6, 5))
            for k in range(5):
                nudge = np.zeros(5)
                nudge[k] = _FD_EPS
                shifted = self._residual(np.clip(qpos + nudge, self._lo, self._hi), target, weight)
                jacobian[:, k] = (shifted - residual) / _FD_EPS
            step = np.linalg.lstsq(
                jacobian.T @ jacobian + _IK_DAMPING * np.eye(5), -jacobian.T @ residual, rcond=None
            )[0]
            norm = float(np.linalg.norm(step))
            if norm > _IK_MAX_STEP:
                step *= _IK_MAX_STEP / norm
            qpos = np.clip(qpos + step, self._lo, self._hi)
            current = float(np.linalg.norm(self._residual(qpos, target, weight)))
            if current < _FD_EPS or abs(previous - current) < 1e-9:
                break
            previous = current
        return qpos

    def measure(
        self, qpos: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        """TCP distance to `target` (m) and approach angle off vertical (deg) at `qpos`."""
        self.forward(qpos)
        error = float(np.linalg.norm(self._data.site_xpos[self._index.tcp_site_id] - target))
        cosine = float(np.clip(self.approach_axis() @ self._down, -1.0, 1.0))
        return error, float(np.degrees(np.arccos(cosine)))

    def solve(
        self,
        target: npt.NDArray[np.float64],
        rng: np.random.Generator,
        weight: float = 1.0,
        accept: Callable[[npt.NDArray[np.float64]], bool] | None = None,
    ) -> tuple[npt.NDArray[np.float64], float, float]:
        """Best configuration found for a world-frame target, with its error and approach angle.

        Seeded first from the resting pose, then from random configurations; returns the best
        attempt whether or not it meets tolerance, so callers can see how far off it was.

        `accept` screens converged, in-tolerance poses for anything this solver does not model —
        self-collision above all. Restarting is what makes it useful: without it the search
        stops at the first pose that reaches the target, which for squares close to the arm is
        routinely one the arm has folded through itself to hold. A rejected pose is never
        returned, so a square with no acceptable solution reports the closest *unacceptable*
        one's error rather than silently offering a pose the caller already refused.
        """
        best_qpos = np.array(self._index.arm_qpos, dtype=np.float64)
        best_error, best_angle = np.inf, np.inf
        restarts = _IK_RESTARTS if accept is None else _IK_SCREENED_RESTARTS
        for attempt in range(restarts):
            seed = (
                np.array(self._index.arm_qpos, dtype=np.float64)
                if attempt == 0
                else rng.uniform(self._lo, self._hi)
            )
            qpos = self._refine(seed, target, weight)
            error, angle = self.measure(qpos, target)
            solved = error <= MAX_TCP_ERROR and angle <= MAX_APPROACH_DEG
            if accept is not None and solved and not accept(qpos):
                continue
            if error + 0.001 * angle < best_error + 0.001 * best_angle:
                best_qpos, best_error, best_angle = qpos, error, angle
            if error <= MAX_TCP_ERROR * 0.1 and angle <= 1.0:
                break
        return best_qpos, best_error, best_angle

    def track(
        self,
        target: npt.NDArray[np.float64],
        seed: npt.NDArray[np.float64],
        weight: float = 1.0,
    ) -> tuple[npt.NDArray[np.float64], float, float]:
        """Refine one fingers-down configuration from `seed`, with no restarts.

        For following a moving target, where the previous configuration is the seed and a
        restart would jump to an unrelated solution between one step and the next.
        """
        qpos = self._refine(np.asarray(seed, dtype=np.float64), target, weight)
        error, angle = self.measure(qpos, target)
        return qpos, error, angle

    def solve_square(
        self,
        square: str,
        board: Board,
        board_pose: BoardPose,
        rng: np.random.Generator,
        height: float = GRASP_HEIGHT,
        accept: Callable[[npt.NDArray[np.float64]], bool] | None = None,
    ) -> GraspPose:
        """Solve a fingers-down pose over one square, `height` above the playing surface."""
        target = board.square_to_world(square, board_pose).astype(np.float64)
        target[2] = board.surface_height + height
        qpos, error, angle = self.solve(target, rng, accept=accept)
        return GraspPose(
            square=square,
            qpos=(qpos[0], qpos[1], qpos[2], qpos[3], qpos[4]),
            tcp_error=error,
            approach_deg=angle,
        )


def load_reference_grasps(path: Path = REFERENCE_GRASPS_PATH) -> dict[str, GraspPose]:
    """The solved fingers-down grasp per square, keyed by square name.

    Squares the arm cannot reach this way are absent rather than present-and-invalid.
    """
    payload = json.loads(path.read_text())
    grasps = {}
    for square, entry in payload["grasps"].items():
        angles = [float(v) for v in entry["qpos"]]
        if len(angles) != len(ARM_JOINT_NAMES):
            raise ValueError(
                f"{path}: grasp for {square} has {len(angles)} joint angles, "
                f"expected {len(ARM_JOINT_NAMES)}"
            )
        grasps[square] = GraspPose(
            square=square,
            qpos=(angles[0], angles[1], angles[2], angles[3], angles[4]),
            tcp_error=float(entry["tcp_error"]),
            approach_deg=float(entry["approach_deg"]),
        )
    return grasps


def reference_metadata(path: Path = REFERENCE_GRASPS_PATH) -> dict[str, object]:
    """The board pose and solver settings the reference grasps were produced under."""
    payload = json.loads(path.read_text())
    metadata: dict[str, object] = payload["metadata"]
    return metadata
