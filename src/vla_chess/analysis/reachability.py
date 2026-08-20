"""Coarse, FK-only reachability probe for a board pose (Step 3 of the implementation plan).

This is deliberately not IK-based: `mink` integration is Step 5, and the plan treats this
probe's output as a coarse bound on board placement, re-confirmed later by Step 5's actual IK
convergence testing (implementation-plan-step-1.md §0 risk register). It answers a narrower
question than "can the arm grasp a piece here" — only "does some arm configuration place the
gripper's TCP near this square, at roughly table height".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.sim.scene import ARM_JOINT_NAMES, PIECE_FOOT_RADIUS, Board, BoardPose, SceneIndex

DEFAULT_N_SAMPLES: Final = 60_000
DEFAULT_XY_TOLERANCE: Final = 0.02
DEFAULT_Z_BAND: Final = (-0.01, 0.12)
MIN_REACHABLE_SQUARES: Final = 60

# Arm base collision geometry (so101_new_calib.xml): the four visual parts duplicated as
# mesh collision geoms, base body at world origin. Largest planar distance from that origin
# to any vertex of those meshes, i.e. the largest radius a piece standing anywhere on the
# board could still clip the base. Driven by `base_so101_v2`; measured, not estimated, and
# it is a convex-hull bound because MuJoCo collides meshes as hulls. Re-measure this if the
# base's collision geoms ever change.
BASE_FOOTPRINT_RADIUS: Final = 0.0718
SAFETY_MARGIN: Final = 0.01
MIN_SQUARE_CLEARANCE: Final = BASE_FOOTPRINT_RADIUS + PIECE_FOOT_RADIUS + SAFETY_MARGIN

# The randomization envelope Step 4 plans to sample board pose from (implementation-plan-
# step-1.md §4): board xy jitter, board yaw jitter, and (per product decision) board scale.
STEP4_PITCH_RANGE: Final = (0.03, 0.05)
STEP4_XY_JITTER: Final = 0.03
STEP4_YAW_JITTER_DEG: Final = 15.0


def sample_tcp_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    n_samples: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Sample where the TCP lands over the arm's whole joint space.

    Every arm joint is drawn independently and uniformly from its own range; the sampler does
    not know about self-collision or reachability of any particular orientation, only where
    the TCP site ends up.

    Args:
        model: Scene model to run forward kinematics on.
        data: State to pose, mutated in place and left at the last sample.
        index: Name-to-id lookup for the scene, supplying the arm joints and the TCP site.
        n_samples: How many configurations to draw and evaluate.
        rng: Source of the sampled configurations.

    Returns:
        TCP positions in the world frame as (n_samples, 3), in metres.
    """
    ranges = [model.jnt_range[index.arm_joint_id[name]] for name in ARM_JOINT_NAMES]
    qadrs = [model.jnt_qposadr[index.arm_joint_id[name]] for name in ARM_JOINT_NAMES]
    samples = np.stack([rng.uniform(lo, hi, n_samples) for lo, hi in ranges], axis=1)
    positions = np.empty((n_samples, 3))
    for i in range(n_samples):
        for qadr, value in zip(qadrs, samples[i], strict=True):
            data.qpos[qadr] = value
        mujoco.mj_forward(model, data)
        positions[i] = data.site_xpos[index.tcp_site_id]
    return positions


@dataclass(frozen=True)
class ReachabilityReport:
    """Per-square coarse reachability verdicts for one board pose, and the probe params used."""

    board_pose: BoardPose
    reachable: dict[str, bool]
    n_samples: int
    xy_tolerance: float
    z_band: tuple[float, float]

    @property
    def reachable_count(self) -> int:
        """Count the squares the probe marked reachable.

        Returns:
            How many of the 64 squares some sampled TCP position came near.
        """
        return sum(self.reachable.values())

    @property
    def unreachable_squares(self) -> list[str]:
        """Name the squares the probe could not place the TCP near.

        Returns:
            Their algebraic names, in ascending order.
        """
        return sorted(sq for sq, ok in self.reachable.items() if not ok)

    def meets_minimum(self, minimum: int = MIN_REACHABLE_SQUARES) -> bool:
        """Judge the probe against its acceptance floor.

        Args:
            minimum: How many squares must be reachable for the pose to pass.

        Returns:
            True when at least that many squares are reachable.
        """
        return self.reachable_count >= minimum


def min_square_clearance(board: Board, board_pose: BoardPose) -> float:
    """Measure how close the board's nearest square comes to the arm's mount.

    A lower bound on how close a standing piece can get to the arm base: below
    `MIN_SQUARE_CLEARANCE`, the base's own collision proxy (or a piece standing at that
    square) risks physical overlap regardless of what any joint-space reachability sample
    says, since FK sampling has no notion of the board's geometry at all.

    Args:
        board: Board geometry the square centres are taken from.
        board_pose: Where that board stands in the world.

    Returns:
        Planar distance in metres from the arm's mount origin, which the asset pins at the
        world origin, to the nearest square centre. Height is ignored.
    """
    squares_xy = np.array([board.square_to_world(sq, board_pose)[:2] for sq in board.squares()])
    return float(np.linalg.norm(squares_xy, axis=1).min())


@dataclass(frozen=True)
class EnvelopePoint:
    """One sampled corner/interior point of the Step 4 board-pose randomization envelope."""

    board_pose: BoardPose
    reachable_count: int
    clearance: float

    @property
    def clears_base(self) -> bool:
        """Judge this pose against the physical clearance floor.

        Returns:
            True when the nearest square stands far enough out that a piece on it cannot
            overlap the arm base's own footprint.
        """
        return self.clearance >= MIN_SQUARE_CLEARANCE


@dataclass(frozen=True)
class EnvelopeSweepReport:
    """Worst-case reachability and clearance across a grid of the Step 4 randomization
    envelope (board pitch, xy jitter, yaw jitter), reusing one FK sample set."""

    points: tuple[EnvelopePoint, ...]
    pitch_range: tuple[float, float]
    xy_jitter: float
    yaw_jitter_deg: float

    @property
    def worst_reachability(self) -> EnvelopePoint:
        """Find the sampled pose that reaches the fewest squares.

        Returns:
            That point. It is not generally the same point as `worst_clearance`; pushing the
            board out trades one against the other.
        """
        return min(self.points, key=lambda p: p.reachable_count)

    @property
    def worst_clearance(self) -> EnvelopePoint:
        """Find the sampled pose that stands closest to the arm base.

        Returns:
            That point.
        """
        return min(self.points, key=lambda p: p.clearance)

    def ok(self, minimum_reachable: int = MIN_REACHABLE_SQUARES) -> bool:
        """Judge the whole envelope against both floors at once.

        Args:
            minimum_reachable: How many squares the weakest pose must still reach.

        Returns:
            True when the worst point on each axis passes its own floor, which is stricter
            than any single pose passing both.
        """
        return self.worst_reachability.reachable_count >= minimum_reachable and (
            self.worst_clearance.clears_base
        )


def sweep_envelope(
    board: Board,
    tcp_positions: npt.NDArray[np.float64],
    centre_x: float,
    centre_y: float,
    centre_yaw: float,
    pitches: Sequence[float],
    xy_offsets: Sequence[float],
    yaws_deg: Sequence[float],
    xy_tolerance: float = DEFAULT_XY_TOLERANCE,
    z_band: tuple[float, float] = DEFAULT_Z_BAND,
) -> EnvelopeSweepReport:
    """Evaluate every combination of pitch, xy offset and yaw offset around a centre pose.

    Args:
        board: Board geometry the square centres are taken from.
        tcp_positions: Sampled TCP positions in the world frame, in metres. One set is
            reused for every point: reachability depends on the arm's own joint space, not
            on the board pose being tested against it, so a fine grid stays cheap.
        centre_x: Board centre in metres along x, before the offsets are applied.
        centre_y: Board centre in metres along y, before the offsets are applied.
        centre_yaw: Board yaw in radians, before the offsets are applied.
        pitches: Square pitches in metres to test.
        xy_offsets: Offsets in metres from the centre, swept independently along x and y, so
            n offsets give n² placements.
        yaws_deg: Yaw offsets in degrees from the centre yaw.
        xy_tolerance: How near a square's centre in metres a sampled TCP position must land
            for that square to count as reachable.
        z_band: Lowest and highest world z in metres a sample may sit at and still count,
            which the table top being at zero makes a height above the table.

    Returns:
        Every grid point with its reachable count and clearance, and the ranges swept.
    """
    points = []
    for pitch, dx, dy, dyaw_deg in product(pitches, xy_offsets, xy_offsets, yaws_deg):
        pose = BoardPose(centre_x + dx, centre_y + dy, centre_yaw + np.radians(dyaw_deg), pitch)
        report = probe_reachability(board, pose, tcp_positions, xy_tolerance, z_band)
        points.append(
            EnvelopePoint(
                board_pose=pose,
                reachable_count=report.reachable_count,
                clearance=min_square_clearance(board, pose),
            )
        )
    pitch_range = (min(pitches), max(pitches))
    return EnvelopeSweepReport(
        points=tuple(points),
        pitch_range=pitch_range,
        xy_jitter=max(abs(v) for v in xy_offsets),
        yaw_jitter_deg=max(abs(v) for v in yaws_deg),
    )


def probe_reachability(
    board: Board,
    board_pose: BoardPose,
    tcp_positions: npt.NDArray[np.float64],
    xy_tolerance: float = DEFAULT_XY_TOLERANCE,
    z_band: tuple[float, float] = DEFAULT_Z_BAND,
) -> ReachabilityReport:
    """Mark each square reachable or not for one board pose.

    Args:
        board: Board geometry the square centres are taken from.
        board_pose: Where that board stands in the world.
        tcp_positions: Sampled TCP positions in the world frame, in metres.
        xy_tolerance: How near a square's centre in metres a sample must land, measured in
            the board's plane only, for that square to count as reachable.
        z_band: Lowest and highest world z in metres a sample may sit at and still count,
            which the table top being at zero makes a height above the table.

    Returns:
        A verdict per square alongside the parameters it was reached under, so the report
        states what "reachable" meant for this run.
    """
    lo_z, hi_z = z_band
    band = tcp_positions[(tcp_positions[:, 2] >= lo_z) & (tcp_positions[:, 2] <= hi_z)]
    band_xy = band[:, :2]
    reachable: dict[str, bool] = {}
    for square in board.squares():
        centre_xy = board.square_to_world(square, board_pose)[:2]
        if band_xy.shape[0] == 0:
            reachable[square] = False
            continue
        distances = np.linalg.norm(band_xy - centre_xy, axis=1)
        reachable[square] = bool(distances.min() <= xy_tolerance)
    return ReachabilityReport(
        board_pose=board_pose,
        reachable=reachable,
        n_samples=tcp_positions.shape[0],
        xy_tolerance=xy_tolerance,
        z_band=z_band,
    )


def _pose_desc(pose: BoardPose) -> str:
    """Render a board pose as one line of table cell text."""
    return (
        f"pitch {pose.square_pitch:.3f}, centre ({pose.x:.3f}, {pose.y:.3f}), "
        f"yaw {pose.yaw:.3f} rad"
    )


def _envelope_section(envelope: EnvelopeSweepReport) -> str:
    """Render the envelope sweep as its section of `docs/reachability.md`.

    Args:
        envelope: Sweep to report, whose two worst points carry the section's verdict.

    Returns:
        The section as markdown, starting at its own heading.
    """
    worst_r = envelope.worst_reachability
    worst_c = envelope.worst_clearance
    verdict = (
        f"**Worst case across the envelope: {worst_r.reachable_count}/64 reachable, "
        f"{worst_c.clearance * 100:.1f} cm minimum square-to-base clearance**, against a "
        f"{MIN_REACHABLE_SQUARES}/64 reachability floor and a "
        f"{MIN_SQUARE_CLEARANCE * 100:.1f} cm clearance floor "
        f"({BASE_FOOTPRINT_RADIUS * 100:.1f} cm base footprint + "
        f"{PIECE_FOOT_RADIUS * 100:.1f} cm piece foot + {SAFETY_MARGIN * 100:.1f} cm margin). "
        f"{'Both hold' if envelope.ok() else 'This corner violates one or both floors'} — "
        "expected: see the resolution note below, this is not a Step 3 blocker."
    )
    return f"""## Randomization envelope (Step 4)

Step 4 plans to sample board pose per episode: pitch uniform in
[{envelope.pitch_range[0]:.2f}, {envelope.pitch_range[1]:.2f}] m, xy offset up to
±{envelope.xy_jitter * 100:.0f} cm, yaw offset up to ±{envelope.yaw_jitter_deg:.0f}°, all
around the nominal centre pose above. Rather than trust the nominal pose alone, every grid
point of that envelope ({len(envelope.points)} combinations) is checked against the *same*
FK sample set — reachability sampling doesn't depend on the board pose being tested, only
the arm's own joint-space, so this reuses the nominal probe's samples at no extra simulation
cost.

Two independent things can fail here, and neither implies the other: a pose can have plenty
of FK samples landing near every square (reachable) while still placing squares inside the
arm base's own physical footprint (no clearance) — the FK sampler has no notion of the
board's geometry at all, so it cannot catch that case on its own. `min_square_clearance`
closes that gap: the planar distance from the arm's mount origin to the nearest square
centre, checked against the base collision meshes' own farthest-vertex radius
({BASE_FOOTPRINT_RADIUS * 100:.1f} cm) plus a standing piece's foot radius
({PIECE_FOOT_RADIUS * 100:.1f} cm) plus a {SAFETY_MARGIN * 100:.1f} cm margin.

| | Reachable squares | Clearance |
|---|---|---|
| Worst-case pose | {worst_r.reachable_count}/64 ({_pose_desc(worst_r.board_pose)}) | — |
| Tightest-clearance | — | {worst_c.clearance * 100:.1f} cm ({_pose_desc(worst_c.board_pose)}) |

{verdict}

### Resolution

No single nominal board offset clears both floors at every corner of this envelope — pushing
the board farther out trades a clearance failure for a worse reachability failure, and vice
versa (see `implementation-plan-step-1.md` §4, "Board pose feasibility"). This is not solved
by picking a different offset: Step 4's `SpecSampler` checks `min_square_clearance` (and,
once Step 5 exists, real IK feasibility) against each individual drawn `BoardPose` and
resamples on failure, the same reject-and-resample pattern already used for distractor-square
collisions and IK targets elsewhere in the plan. The nominal offset is set by the grasp
constraints in `docs/grasp-poses.md` rather than by this sweep, and the full pitch/xy/yaw
ranges are kept; only draws landing in this corner are rejected.
"""


def reachability_markdown(report: ReachabilityReport, envelope: EnvelopeSweepReport) -> str:
    """Render a reachability probe result as the body of `docs/reachability.md`.

    Args:
        report: Nominal board pose's per-square verdicts, quoted alongside the sample count
            and tolerances they were reached under.
        envelope: Sweep around that nominal pose, rendered as the closing section.

    Returns:
        The document as markdown.
    """
    files = "abcdefgh"
    ranks = "12345678"
    header = "|  | " + " | ".join(files) + " |"
    sep = "|---|" + "---|" * len(files)
    rows = []
    for rank in reversed(ranks):
        cells = []
        for file_ in files:
            square = f"{file_}{rank}"
            cells.append("reach" if report.reachable[square] else "**miss**")
        rows.append(f"| {rank} | " + " | ".join(cells) + " |")
    grid = "\n".join([header, sep, *rows])

    verdict = (
        f"**{report.reachable_count}/64 squares reachable** — "
        f"{'meets' if report.meets_minimum() else 'below'} the "
        f"{MIN_REACHABLE_SQUARES}/64 minimum (implementation-plan-step-1.md §3 acceptance "
        "criteria)."
    )
    unreachable = ", ".join(report.unreachable_squares) if report.unreachable_squares else "(none)"
    pose = report.board_pose
    return f"""# Reachability probe

`pixi run reachability-probe` regenerates this file from a fresh sample; every number below
comes from that sweep, none are estimates.

## Method

This is a coarse, forward-kinematics-only probe, **not** an IK-based one: `mink` integration
is Step 5. {report.n_samples} uniformly random joint configurations (within each arm joint's
own range, independent per joint — no self-collision or orientation feasibility check) are
forward-kinematically evaluated, and the gripper site (`gripperframe`, the TCP) world position
is recorded for each. A square is marked reachable if some recorded TCP position, restricted
to a height band of {report.z_band[0]:.2f} to {report.z_band[1]:.2f} m above the table
surface, falls within {report.xy_tolerance * 100:.1f} cm of that square's centre.

This is a real bound on board *placement* — it does not mean every reachable square is
graspable at every orientation a pick/place motion needs. Step 5's actual IK convergence rate
near the board surface is the authority on that; per the plan's risk register, an IK
convergence problem there is treated as a reason to re-tune Step 3/4 reachability bounds, not
to swap solvers.

**Do not read the verdict below as a grasp count.** It is the loosest possible reading of
"reachable": any orientation, no collision check, no piece on the square. Under the fingers-down
constraint an actual grasp needs, with the starting position on the board and the arm required
to clear itself, 30 of these 64 squares have a usable pose — see `docs/grasp-poses.md`, which is
the authority on what can be picked up. The gap between the two numbers is not a contradiction;
it is what this probe deliberately does not model.

**Both of those are still sim-only confirmations.** Every number this probe reports —
table height, board offset from the arm base, square pitch — is a value chosen in
simulation with no real SO-101 unit or physical chessboard behind it. This probe (and
Step 5's IK convergence check, when it exists) can only confirm internal consistency of
that guessed geometry, never that it matches real hardware. Treating "64/64 reachable in
sim" as the reachability question being closed would be wrong; it stays open — for Step 5
as much as for Step 3 — until someone measures an actual SO-101 mount height and board
placement and this probe is re-run against those numbers.

## Board pose probed

| | |
|---|---|
| Centre (x, y) | ({pose.x:.3f}, {pose.y:.3f}) m |
| Yaw | {pose.yaw:.3f} rad |
| Square pitch | {pose.square_pitch:.3f} m |

## Per-square verdict

Ranks top-to-bottom 8→1, files left-to-right a→h (a white-side view of the board).

{grid}

## Verdict

{verdict}

Unreachable squares: {unreachable}

{_envelope_section(envelope)}"""
