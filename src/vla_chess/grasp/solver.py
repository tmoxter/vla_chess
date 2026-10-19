"""Vertical (fingers-down) grasp poses: the IK that finds them, and the solved reference set."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess import paths
from vla_chess.sim.scene import (
    ARM_JOINT_NAMES,
    DEFAULT_BOARD_POSE,
    GRIPPER_BODY_NAME,
    SURFACE_HEIGHT,
    Board,
    BoardPose,
    SceneIndex,
)

REFERENCE_GRASPS_PATH: Final = paths.ASSETS_DIR / "reference_grasps.json"

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

# Relaxed acceptance cap for tilted grasps: how far off vertical the wrist may lean when a
# fingers-down pose is infeasible. Solving position-first (weight 0) and accepting up to this
# angle opens the near-arm block the vertical cap threw away — ~37 fields, ranks 4-centre
# through 8. See experiment-2-plan.md, "Grasp feasibility: vertical is too strict".
MAX_TILT_DEG: Final = 45.0

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
# Restart budget for the min-tilt search. Position-first IK lands on a random tilt from the
# continuum per seed, so the per-square minimum is the lowest over many restarts (the recount
# method). Matched to the value the tilted pick-place prototype was validated at.
_IK_TILT_RESTARTS: Final = 40
_FD_EPS: Final = 1e-6

# How far the roll joint should stay off its own limit. A pose on a stop is one the actuator
# holds by pushing into the stop, with nothing left over for the tracking error the executor
# reports, and the restarts land there often: clipping a random seed's surplus roll costs the
# residual nothing, because roll is the direction the fingers-down constraint leaves free. It is
# a preference rather than a bound — a square whose only solution is clamped still gets one.
MIN_LIMIT_MARGIN: Final = np.radians(2.0)

# How far off the lowest achievable tilt a pose may sit and still be eligible when the caller
# asks for the one nearest a preferred configuration. The min-tilt search returns a random roll
# from the continuum each restart, so without a preference consecutive anchors land in unrelated
# roll branches and the joint-space legs between them unwind the wrist the long way around — a
# full turn when a square accepts a roll near one limit and the survey pose sits near the other.
# A pawn is axially symmetric, so every roll grips it equally: spending a few degrees of extra
# tilt to pick the roll the arm is already near removes the spin at no cost to the grasp.
TILT_ROLL_BAND_DEG: Final = 6.0

_ROLL: Final = ARM_JOINT_NAMES.index("wrist_roll")


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
    max_approach_deg: float = MAX_APPROACH_DEG

    def is_within_tolerance(self) -> bool:
        """Report whether this pose is close enough to grip the square.

        Returns:
            True when the TCP error and the approach angle both sit inside the acceptance
            thresholds this pose was solved under (`max_approach_deg` is the vertical cap for a
            fingers-down pose, or the relaxed tilt cap for a tilted one).
        """
        return self.tcp_error <= MAX_TCP_ERROR and self.approach_deg <= self.max_approach_deg


def finger_axis_site_frame(
    model: mujoco.MjModel, data: mujoco.MjData, index: SceneIndex
) -> npt.NDArray[np.float64]:
    """Measure the direction the fingers stick out in, in the TCP site's own frame.

    Worth measuring rather than assuming: it is very nearly the site's local x, so code that
    reasons about approach direction via the site's z axis constrains the gripper's roll
    instead of its approach, and selects side approaches while reporting them as top-down.

    Args:
        model: Scene model carrying the gripper body and the TCP site.
        data: State forward kinematics is run on; mutated.
        index: Name-to-id lookup for the scene, supplying the TCP site.

    Returns:
        Unit vector in the site's own frame, pointing from the gripper body out along the
        fingers.
    """
    mujoco.mj_kinematics(model, data)
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY_NAME))
    world = data.site_xpos[index.tcp_site_id] - data.xpos[body_id]
    world = world / np.linalg.norm(world)
    rotation = data.site_xmat[index.tcp_site_id].reshape(3, 3)
    return np.asarray(rotation.T @ world, dtype=np.float64)


def _arm_addresses(model: mujoco.MjModel, index: SceneIndex) -> npt.NDArray[np.int64]:
    """Locate the five arm joints in the model's qpos vector, in `ARM_JOINT_NAMES` order."""
    return np.array(
        [model.jnt_qposadr[index.arm_joint_id[name]] for name in ARM_JOINT_NAMES], dtype=np.int64
    )


def joint_limits(
    model: mujoco.MjModel, index: SceneIndex
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read the travel limits of the five arm joints.

    Args:
        model: Scene model carrying the joint ranges.
        index: Name-to-id lookup for the scene, supplying the arm joints.

    Returns:
        Lower and upper bounds in radians, each in `ARM_JOINT_NAMES` order.
    """
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

    Fingers-down is the default. Passing `weight=0` with a relaxed `max_approach_deg` instead
    solves position-first and accepts whatever wrist tilt the square needs up to that cap — the
    tilted-grasp mode that opens the squares no vertical pose reaches.

    Mutates the `data` it is constructed with.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, index: SceneIndex) -> None:
        """Bind the solver to one model and measure the hand it will be solving for.

        Args:
            model: Scene model to solve in.
            data: State every solve writes joint angles into and runs kinematics on; mutated.
            index: Name-to-id lookup for the scene, supplying the arm joints, the TCP site and
                the resting configuration the first seed is taken from.
        """
        self._model = model
        self._data = data
        self._index = index
        self._qadr = _arm_addresses(model, index)
        self._lo, self._hi = joint_limits(model, index)
        self._finger_axis = finger_axis_site_frame(model, data, index)
        self._down = np.array(WORLD_DOWN, dtype=np.float64)

    @property
    def finger_axis(self) -> npt.NDArray[np.float64]:
        """Report the finger axis measured from this model.

        Returns:
            Unit vector in the TCP site's frame, along which the fingers stick out.
        """
        return self._finger_axis

    def forward(self, qpos: npt.NDArray[np.float64]) -> None:
        """Write an arm configuration into the bound state and run forward kinematics.

        Args:
            qpos: Arm configuration in `ARM_JOINT_NAMES` order, radians.
        """
        self._data.qpos[self._qadr] = qpos
        mujoco.mj_kinematics(self._model, self._data)

    def approach_axis(self) -> npt.NDArray[np.float64]:
        """Read the direction the fingers point in at the configuration last written.

        Returns:
            Unit vector in the world frame.
        """
        rotation = self._data.site_xmat[self._index.tcp_site_id].reshape(3, 3)
        return np.asarray(rotation @ self._finger_axis, dtype=np.float64)

    def _residual(
        self, qpos: npt.NDArray[np.float64], target: npt.NDArray[np.float64], weight: float
    ) -> npt.NDArray[np.float64]:
        """Measure how far one configuration is from gripping a point fingers-down.

        Args:
            qpos: Arm configuration to evaluate, in `ARM_JOINT_NAMES` order.
            target: World point the TCP should sit at, metres.
            weight: How hard the approach-axis error counts against the position error.

        Returns:
            Six residuals: the TCP's offset from `target`, then the weighted difference
            between the approach axis and straight down.
        """
        self.forward(qpos)
        position = self._data.site_xpos[self._index.tcp_site_id] - target
        return np.concatenate([position, weight * (self.approach_axis() - self._down)])

    def _refine(
        self, seed: npt.NDArray[np.float64], target: npt.NDArray[np.float64], weight: float
    ) -> npt.NDArray[np.float64]:
        """Descend from one seed to the nearest configuration that grips `target` from above.

        Damped least squares on a finite-difference Jacobian, stopping early once the residual
        stops moving. Every iterate is clipped into the joint limits, so a seed outside them
        still yields a reachable configuration.

        Args:
            seed: Arm configuration to start from, in `ARM_JOINT_NAMES` order.
            target: World point the TCP should sit at, metres.
            weight: How hard the approach-axis error counts against the position error.

        Returns:
            The configuration converged on, which meets no tolerance in particular — how close
            it lands is for the caller to measure.
        """
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

    def free_margin(self, qpos: npt.NDArray[np.float64]) -> float:
        """Measure how far the roll joint stands off its own limit.

        Only that joint: it is the one the residual leaves free, so where it sits is the
        solver's choice and every value it takes has a neighbour a degree away that grasps the
        square just as well. The other four are pinned by the target, and near a stop is often
        the only place they can be — at a fingers-down pose over the arm's home ranks wrist_flex
        runs within 1.5 degrees of its limit whatever else is done.

        Args:
            qpos: Arm configuration to measure, in `ARM_JOINT_NAMES` order.

        Returns:
            Radians from the roll joint to the nearer of its two stops.
        """
        roll = float(np.asarray(qpos, dtype=np.float64)[_ROLL])
        return min(roll - float(self._lo[_ROLL]), float(self._hi[_ROLL]) - roll)

    def travel_cost(
        self, qpos: npt.NDArray[np.float64], prefer: npt.NDArray[np.float64]
    ) -> tuple[int, float, float]:
        """Rank one solved pose against the configuration the arm is coming from.

        Poses on the roll stop sort behind poses off it, then the joint that turns furthest
        decides, then the total. The widest single turn leads because a leg is given a fixed
        duration: it is that joint that sets how fast the arm has to move to hold the schedule.

        Args:
            qpos: Solved arm configuration to rank, in `ARM_JOINT_NAMES` order.
            prefer: Configuration the arm is coming from, in the same order.

        Returns:
            Sort key, cheapest first: whether roll is clamped against a stop, the widest single
            joint turn in radians, then the sum over all five.
        """
        delta = np.abs(qpos - prefer)
        clamped = int(self.free_margin(qpos) < MIN_LIMIT_MARGIN)
        return clamped, float(delta.max()), float(delta.sum())

    def measure(
        self, qpos: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        """Measure how well one configuration grips a target from above.

        Args:
            qpos: Arm configuration to evaluate, in `ARM_JOINT_NAMES` order.
            target: World point the TCP should sit at, metres.

        Returns:
            TCP distance to `target` in metres, and approach angle off vertical in degrees.
        """
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
        prefer: Sequence[float] | None = None,
        max_approach_deg: float = MAX_APPROACH_DEG,
    ) -> tuple[npt.NDArray[np.float64], float, float]:
        """Search for the configuration that best grips a world-frame target from above.

        Seeded first from the resting pose, then from random configurations; returns the best
        attempt whether or not it meets tolerance, so callers can see how far off it was.

        Args:
            target: World point the TCP should sit at, metres.
            rng: Source of the random restart seeds.
            weight: How hard the approach-axis error counts against the position error.
            accept: Screen applied to converged, in-tolerance poses, for anything this solver
                does not model — self-collision above all. Restarting is what makes it useful:
                without it the search stops at the first pose that reaches the target, which
                for squares close to the arm is routinely one the arm has folded through itself
                to hold. A rejected pose is never returned, so a square with no acceptable
                solution reports the closest *unacceptable* one's error rather than silently
                offering a pose the caller already refused.
            prefer: Configuration the arm is coming from. The fingers-down constraint leaves
                roll free, so a square has a continuum of solutions and every one of them
                reaches it equally well; without a preference the first acceptable one wins,
                which is a coin toss between poses a quarter-turn apart. Given one, every
                restart is tried and the closest acceptable pose to it is returned instead
                (see `travel_cost`).
            max_approach_deg: Largest tilt off vertical a pose may hold and still count as
                solved. Defaults to the fingers-down cap; raise it (with `weight=0`) for a
                tilted grasp on a square no vertical pose reaches.

        Returns:
            The chosen configuration in `ARM_JOINT_NAMES` order, its TCP error in metres, and
            its approach angle off vertical in degrees.
        """
        best_qpos = np.array(self._index.arm_qpos, dtype=np.float64)
        best_error, best_angle = np.inf, np.inf
        toward = None if prefer is None else np.asarray(prefer, dtype=np.float64)
        nearest: tuple[tuple[int, float, float], npt.NDArray[np.float64], float, float] | None = (
            None
        )
        restarts = _IK_RESTARTS if accept is None else _IK_SCREENED_RESTARTS
        for attempt in range(restarts):
            seed = (
                np.array(self._index.arm_qpos, dtype=np.float64)
                if attempt == 0
                else rng.uniform(self._lo, self._hi)
            )
            qpos = self._refine(seed, target, weight)
            error, angle = self.measure(qpos, target)
            solved = error <= MAX_TCP_ERROR and angle <= max_approach_deg
            if accept is not None and solved and not accept(qpos):
                continue
            if toward is not None and solved:
                cost = self.travel_cost(qpos, toward)
                if nearest is None or cost < nearest[0]:
                    nearest = (cost, qpos, error, angle)
                continue
            if error + 0.001 * angle < best_error + 0.001 * best_angle:
                best_qpos, best_error, best_angle = qpos, error, angle
            if error <= MAX_TCP_ERROR * 0.1 and angle <= 1.0:
                break
        if nearest is not None:
            return nearest[1], nearest[2], nearest[3]
        return best_qpos, best_error, best_angle

    def solve_min_tilt(
        self,
        target: npt.NDArray[np.float64],
        rng: np.random.Generator,
        accept: Callable[[npt.NDArray[np.float64]], bool] | None = None,
        max_approach_deg: float = MAX_TILT_DEG,
        restarts: int = _IK_TILT_RESTARTS,
        tilt_target_deg: float | None = None,
        prefer: Sequence[float] | None = None,
    ) -> tuple[npt.NDArray[np.float64], float, float, bool]:
        """Search the tilt continuum over a target for the pose at a chosen lean.

        Solves position-first (weight 0) from many restarts, keeping every converged pose that
        reaches the target within tolerance, leans no further than `max_approach_deg`, and comes
        through `accept`. Among those it returns the one whose tilt is lowest, or — when
        `tilt_target_deg` is given — the one whose tilt is nearest that angle, which is how a
        sampler asks for a grasp deliberately leaned past the minimum.

        Each seed lands on a random roll from the continuum, so the lowest-tilt pose carries an
        arbitrary wrist roll. With `prefer` given, the tilt selection is widened to a band around
        the best achievable tilt (`TILT_ROLL_BAND_DEG`) and the pose within it that turns the wrist
        least from `prefer` is returned instead — the same `travel_cost` rule the fingers-down
        `solve` uses, so successive anchors share a roll branch and the joint-space legs between
        them stop spinning the gripper the long way around.

        Args:
            target: World point the TCP should sit at, metres.
            rng: Source of the random restart seeds.
            accept: Screen applied to converged, in-tolerance poses; see `solve`. A pose it
                rejects is never selected.
            max_approach_deg: Largest tilt off vertical a pose may hold and still count.
            restarts: How many seeds to try; the minimum is only as honest as this is large.
            tilt_target_deg: Lean to aim for, degrees off vertical. The nearest accepted pose to
                it is returned; the lowest-tilt accepted pose is returned when omitted.
            prefer: Configuration the arm is coming from. When given, the roll nearest it (within
                `TILT_ROLL_BAND_DEG` of the best tilt) is chosen over the strictly minimum-tilt one.

        Returns:
            The chosen configuration in `ARM_JOINT_NAMES` order, its TCP error in metres, its
            approach angle off vertical in degrees, and whether any pose was accepted at all.
            When none was, the first three describe the closest attempt by position error and the
            flag is False, so the caller can report how far off the square was.
        """
        target = np.asarray(target, dtype=np.float64)
        accepted: list[tuple[float, npt.NDArray[np.float64], float, float]] = []
        best_key = np.inf
        fallback_qpos = np.array(self._index.arm_qpos, dtype=np.float64)
        fallback_err, fallback_angle = np.inf, np.inf
        for attempt in range(restarts):
            seed = (
                np.array(self._index.arm_qpos, dtype=np.float64)
                if attempt == 0
                else rng.uniform(self._lo, self._hi)
            )
            qpos = self._refine(seed, target, weight=0.0)
            error, angle = self.measure(qpos, target)
            if error < fallback_err:
                fallback_qpos, fallback_err, fallback_angle = qpos, error, angle
            if error > MAX_TCP_ERROR or angle > max_approach_deg:
                continue
            if accept is not None and not accept(qpos):
                continue
            key = abs(angle - tilt_target_deg) if tilt_target_deg is not None else angle
            best_key = min(best_key, key)
            accepted.append((key, qpos, error, angle))
        if not accepted:
            return fallback_qpos, fallback_err, fallback_angle, False
        if prefer is None:
            key, qpos, error, angle = min(accepted, key=lambda c: c[0])
            return qpos, error, angle, True
        toward = np.asarray(prefer, dtype=np.float64)
        band = [c for c in accepted if c[0] <= best_key + TILT_ROLL_BAND_DEG]
        _, qpos, error, angle = min(band, key=lambda c: self.travel_cost(c[1], toward))
        return qpos, error, angle, True

    def track(
        self,
        target: npt.NDArray[np.float64],
        seed: npt.NDArray[np.float64],
        weight: float = 1.0,
    ) -> tuple[npt.NDArray[np.float64], float, float]:
        """Refine one fingers-down configuration from `seed`, with no restarts.

        For following a moving target, where the previous configuration is the seed and a
        restart would jump to an unrelated solution between one step and the next.

        Args:
            target: World point the TCP should sit at, metres.
            seed: Arm configuration to start from, in `ARM_JOINT_NAMES` order.
            weight: How hard the approach-axis error counts against the position error.

        Returns:
            The refined configuration, its TCP error in metres, and its approach angle off
            vertical in degrees.
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
        prefer: Sequence[float] | None = None,
        weight: float = 1.0,
        max_approach_deg: float = MAX_APPROACH_DEG,
    ) -> GraspPose:
        """Solve a grasp pose over one square, fingers-down by default.

        Args:
            square: Algebraic square name, e.g. "e4".
            board: Board geometry the square is measured on.
            board_pose: World placement of that board.
            rng: Source of the random restart seeds.
            height: How far above the playing surface the TCP should sit, metres.
            accept: Screen applied to converged, in-tolerance poses; see `solve`.
            prefer: Configuration the arm is coming from; see `solve`.
            weight: How hard the approach-axis error counts against position; see `solve`. Pass
                0 for a tilted grasp so position is met and the wrist takes its natural tilt.
            max_approach_deg: Largest tilt off vertical the pose may hold; see `solve`.

        Returns:
            The pose settled on, carrying the error and approach angle it was reached at and the
            cap it was solved under — in tolerance or not.
        """
        target = board.square_to_world(square, board_pose).astype(np.float64)
        target[2] = board.surface_height + height
        qpos, error, angle = self.solve(
            target,
            rng,
            weight=weight,
            accept=accept,
            prefer=prefer,
            max_approach_deg=max_approach_deg,
        )
        return GraspPose(
            square=square,
            qpos=(qpos[0], qpos[1], qpos[2], qpos[3], qpos[4]),
            tcp_error=error,
            approach_deg=angle,
            max_approach_deg=max_approach_deg,
        )


def load_reference_grasps(path: Path = REFERENCE_GRASPS_PATH) -> dict[str, GraspPose]:
    """Load the solved fingers-down grasp of every square that has one.

    Args:
        path: Reference-grasp JSON to read.

    Returns:
        One pose per square, keyed by square name. Squares the arm cannot reach this way are
        absent rather than present-and-invalid.

    Raises:
        ValueError: If a stored grasp does not carry one angle per arm joint.
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
    """Read the conditions the reference grasps were produced under.

    Args:
        path: Reference-grasp JSON to read.

    Returns:
        The board pose and solver settings recorded alongside the grasps.
    """
    payload = json.loads(path.read_text())
    metadata: dict[str, object] = payload["metadata"]
    return metadata
