"""MuJoCo scene assembly and chess-square/world coordinate mapping."""

from __future__ import annotations

from dataclasses import dataclass
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

# This offset is not a tuning parameter with a smooth optimum: it is pinned between two
# constraints that close on it from opposite sides, and the gap between them is ~3 mm wide.
#
# From below, the arm's own body. Rank 8 is the rank nearest the arm, so d8 and e8 sit right in
# front of the shoulder and the arm has to double back over itself to point the fingers down at
# them. Below x = 0.298 every pose that reaches those two folds the hand back into the shoulder
# or the upper arm, and the black queen and king cannot be lifted off their home squares (see
# docs/grasp-poses.md). That floor moved out from 0.293 when the grasp frame moved to
# `pinchframe`, which is what closed the old 0.293-0.297 window.
#
# From above, the birds-eye survey view. BIRDS_EYE_ARM_QPOS frames all 64 squares from a fixed
# arm pose, so pushing the board out walks the far corners toward the edge of frame. That is a
# property of the searched pose rather than of the arm: the pose committed at 0.295 ran out of
# frame at 0.297, and re-running the search at 0.300 found one with more margin *and* better
# visibility than the pose it replaced (`pixi run search-survey-pose`).
#
# Between those two bounds, 0.300 is where the scene was inspected by eye and signed off, on
# 2026-08-15: the observation views and the back-rank grasps in `docs/reference-views/` are what
# this board distance looks like, and the framing figures recorded in this module are what those
# renders measure. `pixi run render-reference-views` regenerates them.
#
# What does *not* justify moving this is a reach or coverage count. Square-coverage numbers move
# by a few squares between nearby offsets and will happily argue for a placement nobody has ever
# looked at, which is how a second board distance got into the tree once already. If it moves,
# the views have to be re-inspected by eye and every board-dependent figure in this module
# re-derived. Pitch and offset remain unconfirmed guesses (see caveat above), not a measured
# board's dimensions.
DEFAULT_SQUARE_PITCH: Final = 0.05
DEFAULT_BOARD_X: Final = 0.300
DEFAULT_BOARD_Y: Final = 0.0

# The arm plays from behind the black back rank, not from a neutral side seat. The arm's base
# is pinned at the world origin by the vendored asset, so this is expressed by rotating the
# board rather than moving the arm: files a->h run along board-local x and ranks 1->8 along
# board-local y, so a quarter turn puts rank 8 on the near (arm) side and rank 1 on the far
# side, with the a1-h8 diagonal running away to the arm's right.
#
# Rotating a square board a quarter turn about its own centre maps the board onto itself, so
# the 64 square centres keep exactly the same world positions and only their labels move.
# docs/reachability.md's verdicts carry over unchanged rather than needing a re-probe — that
# holds for the jittered envelope too, since yaw jitter about 90 deg is congruent to the same
# jitter about 0 deg.
DEFAULT_BOARD_YAW: Final = np.pi / 2

# Playing-surface height (world z where a piece's base rests): the board base's top face
# plus the square overlay's own half-thickness, both stacked on the tabletop.
#
# The square overlays collide. They are decoration to look at, but making them visual-only puts
# the physical surface 4 mm below this constant, on the board base: pieces then start 4 mm in
# the air, drop on the first step (overshooting to 5.5 mm mid-bounce) and settle where no
# grasp was solved for. Every grasp height in this project is measured from here, so the
# surface the renderer draws and the surface the solver reaches for have to be the same one.
SURFACE_HEIGHT: Final = TABLE_TOP_Z + 2 * BOARD_BASE_HALF_THICKNESS + SQUARE_HALF_THICKNESS

GRIPPER_BODY_NAME: Final = "gripper"

WRIST_CAMERA_NAME: Final = "wrist"
WRIST_CAMERA_BODY: Final = GRIPPER_BODY_NAME


@dataclass(frozen=True)
class WristCameraMount:
    """A wrist-camera mount: position, orientation, and field of view, all in the
    `gripper` body's local frame. Four are provided (`WRIST_CAMERA_MOUNT_OVERHEAD`,
    `WRIST_CAMERA_MOUNT_SIDE`, `WRIST_CAMERA_MOUNT_SIDE_RESTING`, and
    `WRIST_CAMERA_MOUNT_RESTING_OVER_JAWS`) as alternatives to compare empirically, since
    none is obviously better a priori — see the module-level comment above each for the
    tradeoff.

    A mount's orientation is only meaningful together with an arm pose: the camera is
    rigidly attached to the gripper, so `xyaxes` fixes the view relative to the mechanism,
    not relative to the board. `WRIST_CAMERA_MOUNT_RESTING_OVER_JAWS` in particular is
    solved for `RESTING_ARM_QPOS` and is not a general-purpose framing.
    """

    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float


# Mirrors TheRobotStudio/SO-ARM100's own wrist-cam mounts (see `VENDORED.md` and
# `Optional/Wrist_Cam_Mount_*` upstream): bolted on above the wrist-roll body, tipped down
# to look over the fingers at the workspace beyond, rather than straight along the roll
# axis or into the mechanism. `gripperframe` and the moving jaw both sit ~9-10 cm
# along the body's local -z axis, which is also ~world-up when the arm reaches down at the
# board — so the camera's "right" axis is the body's local -y (the arm's own lateral
# direction, stable across reach poses), not local x (the arm's own forward-reach
# direction): aiming "right" along local x made renders look sideways, wiring
# image-horizontal to depth instead of to lateral. "Up" spans local x/z so it tips
# forward-and-up out of the downward view, not sideways.
#
# Tradeoff: the gripper's open/close joint rotates about the body's local y axis (the same
# axis this mount uses as image-right), so the jaw's swing is confined to the local x-z
# plane — the same plane this mount's "up" spans. That makes the jaw's motion project as
# mostly vertical (up/down) in the rendered image, not left-right, no matter how this
# mount's tilt is tuned: right and up are two directions within one shared 2D plane, and
# there is no roll of that plane which puts the swing on "right" without also rotating the
# board out of "upright" by nearly the same amount (verified: at the halfway compromise
# roll, the board was visibly ~45 deg off-level). Board orientation was prioritized here.
WRIST_CAMERA_MOUNT_OVERHEAD: Final = WristCameraMount(
    pos=(0.0, 0.0, 0.14),
    xyaxes=(0.0, -1.0, 0.0, 1.4, 0.0, 0.5),
    fovy=75.0,
)

# Alternative mount: matches the reference hardware photo, which bolts the camera directly
# onto the gripper's own fixed follower piece (`wrist_roll_follower_so101_v1`, the 3D-printed
# body this camera is already attached to) near its distal end, not onto the wrist-roll servo
# housing further back. In the `gripper` body's local frame, the follower piece's own mesh and
# the jaw pivot sit around local z=-0.02 to -0.03, while the wrist-roll servo housing and the
# wrist_flex joint (one body up the chain) sit around local z=+0.02 to +0.06 — so placing the
# camera at negative local z, rather than the positive z used by WRIST_CAMERA_MOUNT_OVERHEAD,
# puts the servo housing and that extra joint behind the camera instead of in frame, matching
# the photo. Right/up axes are unchanged from WRIST_CAMERA_MOUNT_OVERHEAD (already proven to
# read the board as upright); only the position moved forward along the mechanism. Verified
# before picking these numbers: rendered at the hover pose, the servo housing and wrist_flex
# joint are fully out of frame, and open vs. closed jaw renders still show real fingertip
# separation (not just a differently angled static shot).
WRIST_CAMERA_MOUNT_SIDE: Final = WristCameraMount(
    pos=(0.0, 0.03, -0.02),
    xyaxes=(0.0, -1.0, 0.0, 1.4, 0.0, 0.5),
    fovy=75.0,
)

# Same follower-piece placement philosophy as WRIST_CAMERA_MOUNT_SIDE (forward of the
# servo/wrist_flex joint, never behind it), but for the resting/board-reading pose rather
# than close-up move execution. WRIST_CAMERA_MOUNT_SIDE's tight framing is a poor fit there:
# at rest the arm pulls back and the whole board needs to read, not just the jaw's immediate
# neighbourhood. Simply widening WRIST_CAMERA_MOUNT_SIDE's fovy in place doesn't work — at
# that position the lens is close enough to the jaw's own bulk that a wider cone just frames
# more of the mount's own housing, not more board (verified by rendering it: the jaw's mesh
# fills over half the image). This mount adds standoff instead: pushed further out along
# local y and z than WRIST_CAMERA_MOUNT_SIDE so the jaw housing clears the frame, paired with
# a wider fovy and a shallower tilt. Verified before picking these numbers: at the resting
# pose, board coverage is comparable to the original (pre-fix) reference render while the
# servo housing and wrist_flex joint stay out of frame. Trade-off: reused at the close hover
# pose, this framing is too wide for jaw-tracking (the jaw mechanism re-enters frame) — this
# mount is for the resting pose specifically, not a drop-in replacement for
# WRIST_CAMERA_MOUNT_SIDE.
WRIST_CAMERA_MOUNT_SIDE_RESTING: Final = WristCameraMount(
    pos=(0.0, 0.06, -0.05),
    xyaxes=(0.0, -1.0, 0.0, 0.6, 0.0, 1.0),
    fovy=100.0,
)

# The resting mount that pairs with RESTING_ARM_QPOS below: the camera looks out over the
# jaws at the board, so the jaws sit in the lower half of the frame and occlude part of it
# (the centre when closed, spreading toward the corners as they open). That occlusion is
# intended — the policy is expected to move the arm to see what the jaws hide.
#
# Placement follows WRIST_CAMERA_MOUNT_SIDE's rule (forward of the servo housing and the
# wrist_flex joint, so neither is ever in frame) but with 7 cm of standoff along local +y,
# which at RESTING_ARM_QPOS is world-up. Orientation was not hand-tuned: it is a look-at
# construction — forward toward (0.42, 0, SURFACE_HEIGHT) on the board, right = forward x
# world-up, up = right x forward. Deriving right from world-up is what makes the board read
# level *and* the (horizontal) jaw swing read sideways rather than tilted; earlier attempts
# tuned roll by hand and traded one against the other.
#
# Verified by rendering and inspecting, not by numbers alone: board level, both back ranks
# and the two far corners in frame, servo housing and wrist_flex out of frame, and jaws that
# visibly separate left-to-right between fully closed and fully open.
#
# Known limit, structural rather than a tuning miss: the jaw's swing direction is ~42 deg off
# the gripper's lateral axis, tipped along the finger/view axis, so a camera looking out over
# the fingers always sees part of that motion as approach-in-depth. The jaws therefore track
# ~38 deg off image-horizontal (still predominantly sideways). A sweep over gripper position,
# standoff, aim point and fovy bottoms out near 34 deg; only a camera aimed away from the
# board does better, so this is the floor for a board-reading view.
WRIST_CAMERA_MOUNT_RESTING_OVER_JAWS: Final = WristCameraMount(
    pos=(0.0, 0.07, -0.02),
    xyaxes=(1.0, 0.0, -0.000889, -0.000644, 0.689655, -0.724138),
    fovy=90.0,
)

# The over-jaws resting mount is the project's canonical observation view: unless a caller
# deliberately asks for one of the alternatives above, this is what the policy sees. The other
# three mounts are kept as comparison baselines, not as live options.
DEFAULT_WRIST_CAMERA_MOUNT: Final = WRIST_CAMERA_MOUNT_RESTING_OVER_JAWS

ARM_JOINT_NAMES: Final = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# Resting pose: what the arm holds while the policy reads the board, in ARM_JOINT_NAMES order.
#
# The defining constraint is that the jaws' swing plane is parallel to the board, so opening
# and closing reads as lateral motion from the wrist camera. The jaw hinge axis is the
# gripper body's local y, so that constraint is exactly "local y is world-vertical" — and
# because wrist_roll turns about the finger axis (local z), local y can only be brought
# vertical when the finger axis is horizontal. That makes it a solvable equality, not a
# trade-off: this pose holds it to 0.0000 deg with shoulder_pan at zero and no joint against
# a limit, placing the gripper origin at (0.20, 0, 0.16) with the fingers level.
#
# An earlier search reported a hard ~32 deg floor on this. That was wrong: it fixed
# shoulder_lift/elbow_flex/wrist_flex and swept only wrist_roll, so it measured a subspace,
# not the arm.
RESTING_ARM_QPOS: Final = (
    0.0,  # shoulder_pan
    -0.697406345,  # shoulder_lift
    1.203682785,  # elbow_flex
    -0.506276440,  # wrist_flex
    -1.522116821,  # wrist_roll
)
# Birds-eye pose: the survey pose the policy raises to when it needs the whole board, as
# opposed to RESTING_ARM_QPOS which trades coverage for a manipulation-ready stance. Same arm
# root, same board, same camera mount — only joint angles differ.
#
# The camera ends up 0.28 m over the surface looking down at 83 deg, near enough straight down
# that the board reads as a flat grid rather than in perspective. All 64 squares fall inside
# the frame with 6.2% of the half-frame to spare at the worst corner, and 60 of them are also
# unoccluded; the gripper's own jaws cover c7, c8, e7 and f8. Compare the resting pose, which
# has 46 squares in frame and 27 unoccluded.
#
# Framing the board is the constraint here, not the objective. Several poses fit all 64 squares
# and they differ by up to 17 squares in how many the arm's own body then hides, so maximising
# corner margin alone buys headroom with visibility: the widest-margin pose at this distance
# reaches +46% by pulling back to 64 deg, and sees 42 squares against this one's 60. What is
# maximised is squares visible, with the margin held to a 6% floor.
#
# Found by search (`pixi run search-survey-pose`), with two constraints that matter more than
# they look. The horizon is held level by solving for wrist_roll rather than tuning it: with the
# other four joints fixed, the world z-component of the camera's right axis is
# `A cos(roll) + B sin(roll) + C`, so level is a closed-form root, and sampling for it instead
# wastes essentially every draw. Both roots have to be tried — wrist_roll is not the camera's
# optical axis, so one of the two level solutions swings the camera off the board entirely
# rather than merely rolling the image, and taking the first root silently searches a space of
# upward-looking poses. And poses within 0.05 rad of any joint limit were rejected — the
# unconstrained optimum pins wrist_flex against its stop, which leaves the pose no headroom and
# the actuator sitting on a limit.
BIRDS_EYE_ARM_QPOS: Final = (
    0.118154982,  # shoulder_pan
    -1.219742027,  # shoulder_lift
    0.319910277,  # elbow_flex
    1.539854067,  # wrist_flex
    -1.521454882,  # wrist_roll
)

GRIPPER_JOINT_NAME: Final = "gripper"
GRIPPER_SITE_NAME: Final = "gripperframe"

# The gripper closes along the `gripper` body's local x: the fixed finger's face sits at
# x = -0.0079 and the moving jaw sweeps in from positive x. `gripperframe`, the site the
# vendored asset ships, sits *on* that face — it is where the two fingers meet when shut, not
# the centreline between jaws opened wide enough to admit anything. Solving a pose against it
# therefore parks the fixed finger inside the piece: 12.7 mm deep at the d7 reference grasp,
# and screened clean, because the contact check exempted the piece being grasped. The vendored
# site stays in the model and stays untouched; nothing solves against it. See docs/trajectory.md.
#
# The pinch site is that frame shifted along +x, far enough to clear the widest stem this piece
# set presents at grasp height (12 mm, queen and king) with 1.5 mm to spare, so an approach
# never touches the piece and closure brings it to rest against the fixed finger.
#
# The offset is per piece, not one constant for the set, and that was settled by measurement
# rather than taste. A closing jaw drives the piece against the fixed finger, so whatever
# distance separates them at approach is distance the piece gets shoved — and a shove applied
# 25 mm above the board tips a piece at lower force than it slides one. Sized once to the
# widest stem (12 mm) it left a 9 mm pawn stem 4.5 mm of run-up: the d7 pawn was pushed 17 mm
# and toppled into the king on e8, aborting the episode on bystander drift. Sized to the piece
# actually being lifted, the shove is `PINCH_CLEARANCE` and nothing else.
#
# The scene therefore carries the offset (`SceneDescription.pinch_offset`) rather than the
# module: one site, built for the piece that episode moves.
PINCH_CLEARANCE: Final = 0.0015
WIDEST_STEM_RADIUS: Final = 0.012
DEFAULT_PINCH_OFFSET: Final = WIDEST_STEM_RADIUS + PINCH_CLEARANCE
TCP_SITE_NAME: Final = "pinchframe"

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


# The board placement, and the only one. Everything solved against the board — the reference
# grasps in `assets/reference_grasps.json` above all — is solved against this pose, so a second
# "but the grasps were solved at ..." placement must not reappear: poses that hold at one board
# distance miss by tens of millimetres at another, which is small enough to look right in a
# render and large enough to grip air.
DEFAULT_BOARD_POSE: Final = BoardPose(
    DEFAULT_BOARD_X, DEFAULT_BOARD_Y, DEFAULT_BOARD_YAW, DEFAULT_SQUARE_PITCH
)


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


def apply_resting_pose(model: mujoco.MjModel, data: mujoco.MjData, index: SceneIndex) -> None:
    """Drive the arm to the scene's resting pose, leaving the pieces where they stand.

    Takes the target from `index.arm_qpos`, so a scene built with a custom pose rests at that
    pose rather than at the module default. Actuator targets are set alongside the joint
    angles, so a stepped simulation holds the pose instead of drifting out of it on the first
    step. Writes the arm's own addresses only; piece and board state are untouched, and `data`
    is mutated in place.

    Deliberately not routed through the model's qpos0: MuJoCo applies `qpos - qpos0` for hinge
    joints, so seeding qpos0 with these angles would redefine them as the zero rather than move
    the arm to them, leaving the joint values correct and the geometry wrong.
    """
    for name, value in zip(ARM_JOINT_NAMES, index.arm_qpos, strict=True):
        data.qpos[model.jnt_qposadr[index.arm_joint_id[name]]] = value
        data.ctrl[index.arm_actuator_id[name]] = value
    mujoco.mj_forward(model, data)


def make_data(model: mujoco.MjModel, index: SceneIndex) -> mujoco.MjData:
    """A fresh `MjData` for a built scene, with the arm already at the resting pose.

    Use this rather than `mujoco.MjData(model)` directly: a bare `MjData` starts at the arm's
    XML home pose, which is not a pose the policy is ever meant to observe from.
    """
    data = mujoco.MjData(model)
    apply_resting_pose(model, data, index)
    return data


# Jaw gap as a function of the gripper joint angle, over the openings that admit a chess piece
# (qpos -0.02 to +0.16, gaps 14-28 mm). Measured off the vendored meshes rather than derived:
# the jaw face is a printed surface swinging about a hinge, so there is no tidy closed form.
# Linear to within 0.11 mm across that span, comfortably under the squeeze a grip closes by.
JAW_GAP_AT_ZERO: Final = 0.01577
JAW_GAP_PER_RADIAN: Final = 0.07291

# How far past first contact a grip closes. Commanding the mechanical stop instead saturates
# the servo — ~3.35 N·m over a ~77 mm lever is ~43 N of pinch on a 15 g piece — so the command
# is sized to the piece and this is the whole of the overlap it asks for.
GRIP_SQUEEZE: Final = 0.001


def stem_radius(piece_type: PieceType, height: float) -> float:
    """Radius of the geom `piece_type` presents at `height` above its own base."""
    spanning = [
        geom
        for geom in PIECE_GEOMS[piece_type]
        if geom.shape == "cylinder"
        and geom.pos[2] - geom.size[1] <= height <= geom.pos[2] + geom.size[1]
    ]
    if len(spanning) != 1:
        raise ValueError(
            f"{piece_type.value} presents {len(spanning)} cylinders at {height * 1000:.0f} mm, "
            "so it has no single stem to close on there"
        )
    return spanning[0].size[0]


def gripper_qpos_for_gap(gap: float) -> float:
    """Gripper joint angle at which the jaws stand `gap` apart."""
    return (gap - JAW_GAP_AT_ZERO) / JAW_GAP_PER_RADIAN


def grip_command(radius: float, joint_range: tuple[float, float]) -> float:
    """The `[0, 100]` jaw command that closes on a stem of `radius` with a light squeeze."""
    return normalize_gripper(
        gripper_qpos_for_gap(2.0 * radius - GRIP_SQUEEZE),
        joint_range,
    )


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

    board_pose: BoardPose = DEFAULT_BOARD_POSE
    pieces: tuple[PiecePlacement, ...] = ()
    wrist_camera_mount: WristCameraMount = DEFAULT_WRIST_CAMERA_MOUNT
    arm_qpos: tuple[float, float, float, float, float] = RESTING_ARM_QPOS
    pinch_offset: float = DEFAULT_PINCH_OFFSET

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
    gripper_actuator_id: int
    gripper_joint_range: tuple[float, float]
    tcp_site_id: int
    wrist_camera_id: int
    board_body_id: int
    square_geom_id: dict[str, int]
    piece_body_id: dict[str, int]
    arm_qpos: tuple[float, float, float, float, float]


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


def _add_pinch_site(spec: mujoco.MjSpec, offset: float) -> None:
    """Add the pinch site to the gripper body: `gripperframe` shifted along the closing axis.

    Orientation is copied rather than chosen. The grasp solver reads the approach direction off
    this site's own z axis, so rotating it would silently redefine what fingers-down means.
    """
    gripper = spec.body(GRIPPER_BODY_NAME)
    vendored = next(site for site in gripper.sites if site.name == GRIPPER_SITE_NAME)
    pos = np.array(vendored.pos, dtype=float)
    pos[0] += offset
    gripper.add_site(
        name=TCP_SITE_NAME,
        pos=pos.tolist(),
        quat=np.array(vendored.quat, dtype=float).tolist(),
        group=int(vendored.group),
    )


def build_scene(desc: SceneDescription) -> tuple[mujoco.MjModel, SceneIndex]:
    """Assemble the arm + table + board + pieces scene described by `desc`."""
    spec = mujoco.MjSpec.from_file(str(SO101_XML_PATH))
    spec.visual.global_.offwidth = RENDER_OFFWIDTH
    spec.visual.global_.offheight = RENDER_OFFHEIGHT
    _add_pinch_site(spec, desc.pinch_offset)

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
            )
            square_geom_names[square] = geom_name

    piece_body_names: dict[str, str] = {}
    for placement in desc.pieces:
        pos = board.square_to_world(placement.square, board_pose)
        name = f"piece_{placement.square}"
        _add_piece(worldbody, name, placement, pos)
        piece_body_names[placement.square] = name

    mount = desc.wrist_camera_mount
    gripper_body = spec.body(WRIST_CAMERA_BODY)
    gripper_body.add_camera(
        name=WRIST_CAMERA_NAME,
        pos=list(mount.pos),
        xyaxes=list(mount.xyaxes),
        fovy=mount.fovy,
    )

    model = spec.compile()
    index = _build_scene_index(
        model, board, board_pose, square_geom_names, piece_body_names, desc.arm_qpos
    )

    return model, index


def _build_scene_index(
    model: mujoco.MjModel,
    board: Board,
    board_pose: BoardPose,
    square_geom_names: dict[str, str],
    piece_body_names: dict[str, str],
    arm_qpos: tuple[float, float, float, float, float],
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
        gripper_actuator_id=actuator_id(GRIPPER_JOINT_NAME),
        gripper_joint_range=(float(gripper_range[0]), float(gripper_range[1])),
        tcp_site_id=site_id(TCP_SITE_NAME),
        wrist_camera_id=camera_id(WRIST_CAMERA_NAME),
        board_body_id=body_id("board"),
        square_geom_id={sq: geom_id(nm) for sq, nm in square_geom_names.items()},
        piece_body_id={sq: body_id(nm) for sq, nm in piece_body_names.items()},
        arm_qpos=arm_qpos,
    )
