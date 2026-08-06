"""MuJoCo scene assembly and chess-square/world coordinate mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

ASSETS_DIR: Final = Path(__file__).resolve().parent / "assets"
SO101_XML_PATH: Final = ASSETS_DIR / "so101" / "so101_new_calib.xml"

RENDER_OFFWIDTH: Final = 512
RENDER_OFFHEIGHT: Final = 512

# World frame: z=0 is the tabletop. The arm's `base` body sits at the world origin (fixed
# by the vendored asset), so everything else is placed relative to that same surface.
#
# UNCONFIRMED BY REAL HARDWARE: TABLE_TOP_Z, TABLE_HALF_THICKNESS, and every board-placement
# constant below were chosen in simulation only — no measured SO-101 mount height or physical
# chessboard exists behind any of them. docs/reachability.md's "64/64 reachable" verdict is
# internal consistency of this guessed geometry, not confirmation it matches real hardware;
# that stays open through Step 5's IK check too, and is only closed by measuring an actual
# arm/board/table and re-running the probe against those numbers.
TABLE_TOP_Z: Final = 0.0
TABLE_HALF_THICKNESS: Final = 0.18
TABLE_HALF_EXTENT_X: Final = 0.35
TABLE_HALF_EXTENT_Y: Final = 0.35
TABLE_CENTER_X: Final = 0.2
TABLE_RGBA: Final = (0.55, 0.42, 0.30, 1.0)

BOARD_BASE_HALF_THICKNESS: Final = 0.008
SQUARE_HALF_THICKNESS: Final = 0.004
SQUARE_MARGIN: Final = 0.002
BOARD_BORDER: Final = 0.01
BOARD_BASE_RGBA: Final = (0.30, 0.20, 0.14, 1.0)
LIGHT_SQUARE_RGBA: Final = (0.92, 0.88, 0.78, 1.0)
DARK_SQUARE_RGBA: Final = (0.35, 0.24, 0.16, 1.0)

# The chosen default board pose was picked by a coarse geometric reachability probe (see
# docs/reachability.md): centred at this offset with a standard 0.05 m pitch, all 64 squares
# fall inside the arm's sampled reach envelope at near-table height. Pitch and offset are
# themselves unconfirmed guesses (see caveat above), not a measured board's dimensions.
DEFAULT_SQUARE_PITCH: Final = 0.05
DEFAULT_BOARD_X: Final = 0.28
DEFAULT_BOARD_Y: Final = 0.0
DEFAULT_BOARD_YAW: Final = 0.0

# Playing-surface height (world z where a piece's base rests): the board base's top face
# plus the square overlay's own half-thickness, both stacked on the tabletop.
SURFACE_HEIGHT: Final = TABLE_TOP_Z + 2 * BOARD_BASE_HALF_THICKNESS + SQUARE_HALF_THICKNESS

WRIST_CAMERA_NAME: Final = "wrist"
WRIST_CAMERA_BODY: Final = "gripper"
# Offset to the side of the gripper's own motor housing (see docs/reachability.md for how
# this was chosen) so the camera looks past the mechanism at the workspace below, rather
# than straight into it.
WRIST_CAMERA_POS: Final = (0.0, 0.045, -0.01)
WRIST_CAMERA_XYAXES: Final = (1.0, 0.0, 0.0, 0.0, 0.3, 1.0)
WRIST_CAMERA_FOVY: Final = 65.0

ARM_JOINT_NAMES: Final = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT_NAME: Final = "gripper"
GRIPPER_SITE_NAME: Final = "gripperframe"

FILES: Final = "abcdefgh"
RANKS: Final = "12345678"


class PieceType(Enum):
    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"


class PieceColor(Enum):
    WHITE = "white"
    BLACK = "black"


PIECE_COLOR_RGBA: Final = {
    PieceColor.WHITE: (0.90, 0.88, 0.82, 1.0),
    PieceColor.BLACK: (0.12, 0.11, 0.10, 1.0),
}


@dataclass(frozen=True)
class PiecePlacement:
    square: str
    piece_type: PieceType
    color: PieceColor


@dataclass(frozen=True)
class BoardPose:
    """World-frame placement of a board: centre position, yaw, and square pitch."""

    x: float
    y: float
    yaw: float
    square_pitch: float = DEFAULT_SQUARE_PITCH

    def __post_init__(self) -> None:
        if self.square_pitch <= 0.0:
            raise ValueError(f"square_pitch must be positive, got {self.square_pitch}")


def _parse_square(square: str) -> tuple[int, int]:
    if len(square) != 2 or square[0] not in FILES or square[1] not in RANKS:
        raise ValueError(f"not a valid square name: {square!r}")
    return FILES.index(square[0]), RANKS.index(square[1])


def _rotate2d(v: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    result: npt.NDArray[np.float64] = rotation @ v
    return result


class Board:
    """8x8 chess board algebra: square names <-> world-frame poses.

    Pure coordinate math, independent of any MuJoCo model. Files (a..h) run along a board's
    local x axis and ranks (1..8) along its local y axis, both centred on the board pose.
    """

    def __init__(self, surface_height: float = SURFACE_HEIGHT) -> None:
        self._surface_height = surface_height

    @property
    def surface_height(self) -> float:
        return self._surface_height

    def squares(self) -> list[str]:
        """All 64 square names, `"a1"` through `"h8"`."""
        return [f"{f}{r}" for f in FILES for r in RANKS]

    def square_to_world(self, square: str, board: BoardPose) -> npt.NDArray[np.float64]:
        """World-frame centre of `square`'s top surface under the given board pose."""
        file_index, rank_index = _parse_square(square)
        local = np.array(
            [(file_index - 3.5) * board.square_pitch, (rank_index - 3.5) * board.square_pitch]
        )
        world_xy = _rotate2d(local, board.yaw) + np.array([board.x, board.y])
        return np.array([world_xy[0], world_xy[1], self._surface_height])

    def world_to_square(self, xyz: npt.ArrayLike, board: BoardPose) -> str | None:
        """The square whose centre `xyz` (or a point over it) is nearest to, in-plane.

        Ignores the z coordinate. Returns `None` if the point falls outside the 8x8 grid.
        """
        xy = np.asarray(xyz, dtype=np.float64)[:2]
        local = _rotate2d(xy - np.array([board.x, board.y]), -board.yaw)
        file_index = round(float(local[0] / board.square_pitch + 3.5))
        rank_index = round(float(local[1] / board.square_pitch + 3.5))
        if not (0 <= file_index < 8 and 0 <= rank_index < 8):
            return None
        return f"{FILES[file_index]}{RANKS[rank_index]}"


def normalize_gripper(qpos: float, joint_range: tuple[float, float]) -> float:
    """Map a raw gripper joint angle (radians) to the `[0, 100]` closed/open convention."""
    lo, hi = joint_range
    return (qpos - lo) / (hi - lo) * 100.0


def denormalize_gripper(value: float, joint_range: tuple[float, float]) -> float:
    """Map a `[0, 100]` closed/open gripper value back to a raw joint angle (radians)."""
    lo, hi = joint_range
    return lo + (value / 100.0) * (hi - lo)


@dataclass(frozen=True)
class SceneDescription:
    """Everything needed to assemble one MuJoCo scene: board placement and piece layout."""

    board_pose: BoardPose = field(
        default_factory=lambda: BoardPose(
            DEFAULT_BOARD_X, DEFAULT_BOARD_Y, DEFAULT_BOARD_YAW, DEFAULT_SQUARE_PITCH
        )
    )
    pieces: tuple[PiecePlacement, ...] = ()

    def __post_init__(self) -> None:
        squares = [p.square for p in self.pieces]
        duplicates = sorted({s for s in squares if squares.count(s) > 1})
        if duplicates:
            raise ValueError(f"more than one piece placed on the same square: {duplicates}")


@dataclass(frozen=True)
class SceneIndex:
    """Name -> id lookups into a model built by `build_scene`."""

    board: Board
    board_pose: BoardPose
    arm_joint_id: dict[str, int]
    arm_actuator_id: dict[str, int]
    gripper_joint_id: int
    gripper_joint_range: tuple[float, float]
    gripper_site_id: int
    wrist_camera_id: int
    board_body_id: int
    square_geom_id: dict[str, int]
    piece_body_id: dict[str, int]


_GEOM_TYPES: Final = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


@dataclass(frozen=True)
class _PieceGeom:
    shape: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    mass: float
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


# All pieces share a squat cylindrical foot for a stable, flat-bottomed base, then a
# type-specific stack of convex primitives on top. Heights and masses follow Staunton-set
# proportions scaled to DEFAULT_SQUARE_PITCH, not any single real chess set.
PIECE_FOOT_RADIUS: Final = 0.013
_FOOT: Final = _PieceGeom("cylinder", (PIECE_FOOT_RADIUS, 0.006, 0.0), (0.0, 0.0, 0.006), 0.006)

PIECE_GEOMS: Final[dict[PieceType, tuple[_PieceGeom, ...]]] = {
    PieceType.PAWN: (
        _FOOT,
        _PieceGeom("cylinder", (0.009, 0.010, 0.0), (0.0, 0.0, 0.022), 0.006),
        _PieceGeom("sphere", (0.010, 0.0, 0.0), (0.0, 0.0, 0.042), 0.003),
    ),
    PieceType.ROOK: (
        _FOOT,
        _PieceGeom("cylinder", (0.011, 0.012, 0.0), (0.0, 0.0, 0.024), 0.010),
        _PieceGeom("cylinder", (0.0125, 0.004, 0.0), (0.0, 0.0, 0.040), 0.004),
    ),
    PieceType.KNIGHT: (
        _FOOT,
        _PieceGeom("cylinder", (0.010, 0.012, 0.0), (0.0, 0.0, 0.024), 0.009),
        _PieceGeom(
            "box",
            (0.007, 0.014, 0.012),
            (0.0, 0.006, 0.048),
            0.005,
            (0.9659, -0.2588, 0.0, 0.0),
        ),
    ),
    PieceType.BISHOP: (
        _FOOT,
        _PieceGeom("cylinder", (0.010, 0.014, 0.0), (0.0, 0.0, 0.026), 0.011),
        _PieceGeom("capsule", (0.006, 0.010, 0.0), (0.0, 0.0, 0.050), 0.003),
        _PieceGeom("sphere", (0.005, 0.0, 0.0), (0.0, 0.0, 0.065), 0.001),
    ),
    PieceType.QUEEN: (
        _FOOT,
        _PieceGeom("cylinder", (0.012, 0.018, 0.0), (0.0, 0.0, 0.030), 0.016),
        _PieceGeom("cylinder", (0.008, 0.008, 0.0), (0.0, 0.0, 0.056), 0.004),
        _PieceGeom("sphere", (0.011, 0.0, 0.0), (0.0, 0.0, 0.075), 0.005),
    ),
    PieceType.KING: (
        _FOOT,
        _PieceGeom("cylinder", (0.012, 0.020, 0.0), (0.0, 0.0, 0.032), 0.018),
        _PieceGeom("cylinder", (0.008, 0.010, 0.0), (0.0, 0.0, 0.062), 0.005),
        _PieceGeom("box", (0.003, 0.003, 0.010), (0.0, 0.0, 0.082), 0.001),
        _PieceGeom("box", (0.008, 0.003, 0.003), (0.0, 0.0, 0.089), 0.001),
    ),
}


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0)))


def _add_piece(
    parent: mujoco.MjsBody,
    name: str,
    placement: PiecePlacement,
    pos: npt.NDArray[np.float64],
) -> mujoco.MjsBody:
    body = parent.add_body(name=name, pos=list(pos))
    body.add_freejoint()
    rgba = PIECE_COLOR_RGBA[placement.color]
    for i, geom in enumerate(PIECE_GEOMS[placement.piece_type]):
        body.add_geom(
            name=f"{name}_g{i}",
            type=_GEOM_TYPES[geom.shape],
            size=list(geom.size),
            pos=list(geom.pos),
            quat=list(geom.quat),
            mass=geom.mass,
            rgba=list(rgba),
        )
    return body


def build_scene(desc: SceneDescription) -> tuple[mujoco.MjModel, SceneIndex]:
    """Assemble the arm + table + board + pieces scene described by `desc`."""
    spec = mujoco.MjSpec.from_file(str(SO101_XML_PATH))
    spec.visual.global_.offwidth = RENDER_OFFWIDTH
    spec.visual.global_.offheight = RENDER_OFFHEIGHT

    worldbody = spec.worldbody
    worldbody.add_light(
        name="key", pos=[0.35, -0.35, 1.2], dir=[-0.3, 0.3, -1.0], diffuse=[0.7, 0.7, 0.7]
    )
    worldbody.add_light(
        name="fill", pos=[-0.2, 0.4, 1.0], dir=[0.4, -0.4, -1.0], diffuse=[0.25, 0.25, 0.25]
    )

    table = worldbody.add_body(
        name="table", pos=[TABLE_CENTER_X, 0.0, TABLE_TOP_Z - TABLE_HALF_THICKNESS]
    )
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TABLE_HALF_EXTENT_X, TABLE_HALF_EXTENT_Y, TABLE_HALF_THICKNESS],
        rgba=list(TABLE_RGBA),
    )

    board_pose = desc.board_pose
    board_body = worldbody.add_body(
        name="board",
        pos=[board_pose.x, board_pose.y, TABLE_TOP_Z + BOARD_BASE_HALF_THICKNESS],
        quat=list(_yaw_quat(board_pose.yaw)),
    )
    board_half_extent = 4.0 * board_pose.square_pitch + BOARD_BORDER
    board_body.add_geom(
        name="board_base",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[board_half_extent, board_half_extent, BOARD_BASE_HALF_THICKNESS],
        rgba=list(BOARD_BASE_RGBA),
    )

    board = Board()
    square_geom_names: dict[str, str] = {}
    for file_index, file_name in enumerate(FILES):
        for rank_index, rank_name in enumerate(RANKS):
            square = f"{file_name}{rank_name}"
            geom_name = f"sq_{square}"
            half_square = board_pose.square_pitch / 2.0 - SQUARE_MARGIN
            dark = (file_index + rank_index) % 2 == 0
            board_body.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_square, half_square, SQUARE_HALF_THICKNESS],
                pos=[
                    (file_index - 3.5) * board_pose.square_pitch,
                    (rank_index - 3.5) * board_pose.square_pitch,
                    BOARD_BASE_HALF_THICKNESS,
                ],
                rgba=list(DARK_SQUARE_RGBA if dark else LIGHT_SQUARE_RGBA),
                contype=0,
                conaffinity=0,
            )
            square_geom_names[square] = geom_name

    piece_body_names: dict[str, str] = {}
    for placement in desc.pieces:
        pos = board.square_to_world(placement.square, board_pose)
        name = f"piece_{placement.square}"
        _add_piece(worldbody, name, placement, pos)
        piece_body_names[placement.square] = name

    gripper_body = spec.body(WRIST_CAMERA_BODY)
    gripper_body.add_camera(
        name=WRIST_CAMERA_NAME,
        pos=list(WRIST_CAMERA_POS),
        xyaxes=list(WRIST_CAMERA_XYAXES),
        fovy=WRIST_CAMERA_FOVY,
    )

    model = spec.compile()
    index = _build_scene_index(model, board, board_pose, square_geom_names, piece_body_names)
    return model, index


def _build_scene_index(
    model: mujoco.MjModel,
    board: Board,
    board_pose: BoardPose,
    square_geom_names: dict[str, str],
    piece_body_names: dict[str, str],
) -> SceneIndex:
    def body_id(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    def joint_id(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))

    def actuator_id(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))

    def geom_id(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))

    def site_id(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))

    def camera_id(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name))

    gripper_joint = joint_id(GRIPPER_JOINT_NAME)
    gripper_range = model.jnt_range[gripper_joint]

    return SceneIndex(
        board=board,
        board_pose=board_pose,
        arm_joint_id={name: joint_id(name) for name in ARM_JOINT_NAMES},
        arm_actuator_id={name: actuator_id(name) for name in ARM_JOINT_NAMES},
        gripper_joint_id=gripper_joint,
        gripper_joint_range=(float(gripper_range[0]), float(gripper_range[1])),
        gripper_site_id=site_id(GRIPPER_SITE_NAME),
        wrist_camera_id=camera_id(WRIST_CAMERA_NAME),
        board_body_id=body_id("board"),
        square_geom_id={sq: geom_id(nm) for sq, nm in square_geom_names.items()},
        piece_body_id={sq: body_id(nm) for sq, nm in piece_body_names.items()},
    )
