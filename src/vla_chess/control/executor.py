"""Execution of a waypoint plan against the simulator: control loop, aborts, and diagnostics."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.control.planner import Phase, Plan
from vla_chess.grasp.solver import VerticalGraspSolver
from vla_chess.sim.collision import grip_contacts
from vla_chess.sim.scene import (
    ARM_JOINT_NAMES,
    SceneIndex,
    denormalize_gripper,
    normalize_gripper,
)

CONTROL_HZ: Final = 50.0

# A solve this far off the interpolated target is treated as the solver having lost the path
# rather than as tracking error: the anchor poses are exact, so millimetres here mean the
# fingers-down constraint stopped being satisfiable somewhere between them. This is the abort for
# a fingers-down plan, whose Cartesian legs track exactly; a tilted plan grazes further and passes
# `TILTED_IK_ABORT` to `run` instead, letting physics rather than the residual judge the outcome.
MAX_IK_ERROR: Final = 0.005

# The abort a sample-then-filter plan runs under. A tilted Cartesian leg tracks its eased approach
# axis at a fraction of full weight, so it drifts tens of millimetres off the interpolated position
# and still places the piece; this guard only kills a truly diverging rollout fast, and the
# post-hoc validator decides whether what landed is a keep. Measured: clean tilted placements graze
# ~20-24 mm, so 30 mm passes them while stopping a solve that has lost the arm entirely.
TILTED_IK_ABORT: Final = 0.030

# How far a piece that is not the one being moved may shift before the episode is a failure.
MAX_BYSTANDER_DRIFT: Final = 0.005

# How hard a tilted leg's IK tracks its approach axis against its position target. Full weight
# holds the lean exactly but trades away millimetres of position at poses where the eased axis is
# not perfectly achievable; a fraction guides the wrist enough to stay on the tilted IK branch
# while leaving position near-exact. Only tilted legs use it — fingers-down legs track at the
# solver's own default, which the vertical grasps are exact at.
TILT_TRACK_WEIGHT: Final = 0.3

# A stuck or diverging rollout has to end by itself rather than run until someone notices. The
# budget is the executor's own work: capturing observations happens inside this loop through
# `on_control_step`, and rendering two cameras instead of one is the caller's cost, not a sim
# that stopped progressing. Timing it in would have turned "record a witness view as well" into
# an abort halfway through the episode, which is exactly what it did before this was split out.
WALL_CLOCK_LIMIT: Final = 600.0


class AbortReason(Enum):
    NON_FINITE_STATE = "non_finite_state"
    JOINT_LIMIT = "joint_limit"
    IK_DIVERGED = "ik_diverged"
    PIECE_DISTURBED = "piece_disturbed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ExecutionDiagnostics:
    """What the run cost and how well it tracked, whether or not it succeeded."""

    max_tracking_error: float
    max_ik_error: float
    control_steps: int
    sim_steps: int
    wall_seconds: float


@dataclass(frozen=True)
class PlacementCheck:
    """Where a moved piece ended up, and whether that counts as a placement.

    The verdict a closed-loop rollout is scored on: the policy drove the arm for a fixed number
    of frames with no abort bookkeeping, so success is purely "did the piece come to rest, upright,
    on its target square".
    """

    final_square: str | None
    settled: bool
    placed: bool


@dataclass(frozen=True)
class ExecutionResult:
    """One executed rollout: what was commanded, what happened, and how it ended."""

    states: npt.NDArray[np.float64]
    actions: npt.NDArray[np.float64]
    phases: tuple[Phase, ...]
    phase_boundaries: dict[Phase, tuple[int, int]]
    grasped: bool
    placed: bool
    final_square: str | None
    abort_reason: AbortReason | None
    diagnostics: ExecutionDiagnostics

    @property
    def succeeded(self) -> bool:
        """Report whether the rollout did what it set out to do.

        Returns:
            True when the run was never aborted and the piece was both gripped and left
            standing on its destination.
        """
        return self.abort_reason is None and self.grasped and self.placed


def minimum_jerk(alpha: float) -> float:
    """Ease a leg's progress so it starts and ends at rest.

    Args:
        alpha: How far through the leg the motion is, from 0 to 1.

    Returns:
        The eased fraction, with zero velocity and acceleration at both ends.
    """
    return float(alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2))


class Executor:
    """Runs a plan: interpolates each leg, solves IK for the Cartesian ones, steps physics.

    Holds its own scratch `MjData` for kinematics queries. `VerticalGraspSolver` writes joint
    angles into the data it is given and runs forward kinematics on it, so pointing it at the
    simulated state would rewrite the very state being stepped.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        index: SceneIndex,
        control_hz: float = CONTROL_HZ,
    ) -> None:
        """Bind the executor to one model and work out how often to step physics.

        Args:
            model: Scene model to run in, whose timestep sets the decimation.
            index: Name-to-id lookup for the scene, supplying the arm joints, the actuators
                and the pieces.
            control_hz: How often a new command is written, in hertz.

        Raises:
            ValueError: If the control rate is not positive, or is faster than the model's own
                physics timestep.
        """
        if control_hz <= 0.0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        decimation = round(1.0 / (control_hz * float(model.opt.timestep)))
        if decimation < 1:
            raise ValueError(
                f"control rate {control_hz} Hz is faster than the physics timestep "
                f"{model.opt.timestep} s"
            )
        self._model = model
        self._index = index
        self._control_hz = control_hz
        self._decimation = decimation
        self._scratch = mujoco.MjData(model)
        self._solver = VerticalGraspSolver(model, self._scratch, index)
        # The fingers-down axis every Cartesian leg tracks toward unless the waypoint names its
        # own (a tilted grasp). Captured so a tilted leg can be restored to it afterwards.
        self._default_down = np.array(self._solver._down, dtype=np.float64, copy=True)
        self._qadr = np.array(
            [model.jnt_qposadr[index.arm_joint_id[name]] for name in ARM_JOINT_NAMES],
            dtype=np.int64,
        )
        self._gripper_qadr = int(model.jnt_qposadr[index.gripper_joint_id])
        self._lo = np.array(
            [model.jnt_range[index.arm_joint_id[name]][0] for name in ARM_JOINT_NAMES]
        )
        self._hi = np.array(
            [model.jnt_range[index.arm_joint_id[name]][1] for name in ARM_JOINT_NAMES]
        )

    @property
    def control_hz(self) -> float:
        """Report how often a new command is written.

        Returns:
            The control rate in hertz.
        """
        return self._control_hz

    @property
    def decimation(self) -> int:
        """Report how much physics runs between commands.

        Returns:
            Physics steps taken per control step.
        """
        return self._decimation

    def tcp_position(self, qpos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Locate the TCP at an arm configuration, without touching simulated state.

        Args:
            qpos: Arm configuration to measure, in `ARM_JOINT_NAMES` order.

        Returns:
            World position of the TCP in metres, as a copy that later queries will not
            overwrite.
        """
        self._solver.forward(np.asarray(qpos, dtype=np.float64))
        return np.array(self._scratch.site_xpos[self._index.tcp_site_id], copy=True)

    def _arm_qpos(self, data: mujoco.MjData) -> npt.NDArray[np.float64]:
        """Read the five arm joints out of a state, in `ARM_JOINT_NAMES` order."""
        return np.asarray(data.qpos[self._qadr], dtype=np.float64)

    def _approach_axis(self, qpos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Measure the direction the fingers point in at an arm configuration.

        Args:
            qpos: Arm configuration to measure, in `ARM_JOINT_NAMES` order.

        Returns:
            Unit vector in the world frame, used as the start of a tilted leg's eased axis so
            the wrist rotates from where it is rather than snapping to the target lean.
        """
        self._solver.forward(np.asarray(qpos, dtype=np.float64))
        return self._solver.approach_axis()

    def _state(self, data: mujoco.MjData) -> npt.NDArray[np.float64]:
        """Read one recorded state out of the simulator.

        Args:
            data: Stepped state to read.

        Returns:
            Six numbers: the five arm joints in radians, then the jaw on its [0, 100]
            closed-to-open scale.
        """
        gripper = normalize_gripper(
            float(data.qpos[self._gripper_qadr]), self._index.gripper_joint_range
        )
        return np.concatenate([self._arm_qpos(data), [gripper]])

    def _write_ctrl(
        self, data: mujoco.MjData, arm: npt.NDArray[np.float64], gripper: float
    ) -> None:
        """Command one arm configuration and jaw opening.

        Args:
            data: State to write the actuator commands into.
            arm: Joint angles to command, in `ARM_JOINT_NAMES` order.
            gripper: Jaw command on the [0, 100] closed-to-open scale, converted back into a
                joint angle on the way in.
        """
        for name, value in zip(ARM_JOINT_NAMES, arm, strict=True):
            data.ctrl[self._index.arm_actuator_id[name]] = value
        data.ctrl[self._index.gripper_actuator_id] = denormalize_gripper(
            gripper, self._index.gripper_joint_range
        )

    def _holding(self, data: mujoco.MjData, piece_body: int) -> bool:
        """Report whether both jaws are currently bearing on the piece being moved.

        One jaw is not a hold. The fixed finger rides on the `gripper` body and touches for
        reasons unrelated to gripping, so accepting either jaw reported a grasp for the run
        that left the pawn standing on its square (docs/trajectory.md).

        Args:
            data: Stepped state to read the contacts from.
            piece_body: Body id of the piece being moved.

        Returns:
            True when the fixed finger and the moving jaw both bear on it.
        """
        return grip_contacts(self._model, data, piece_body).is_gripped()

    def run(
        self,
        data: mujoco.MjData,
        plan: Plan,
        on_control_step: Callable[[int, mujoco.MjData], None] | None = None,
        wall_clock_limit: float = WALL_CLOCK_LIMIT,
        ik_abort: float = MAX_IK_ERROR,
    ) -> ExecutionResult:
        """Execute a plan, recording one state/action pair per control step.

        Legs are eased between their endpoints and flown either in joint space or, for the
        Cartesian ones, by tracking the interpolated TCP target with fingers-down IK. The run
        ends early on the first abort condition, leaving everything recorded up to that step.

        Args:
            data: Simulated state to run the plan in; stepped throughout.
            plan: The waypoints to fly, in order.
            on_control_step: Called after each step with the step's index and the stepped
                state. It is how observation capture attaches without this loop knowing about
                cameras, and the time it spends is excluded from the wall-clock budget.
            wall_clock_limit: Seconds of the executor's own work allowed before the run is
                aborted as stuck.
            ik_abort: How far a Cartesian leg's solve may sit off its interpolated target before
                the run is aborted as diverged. Defaults to the fingers-down `MAX_IK_ERROR`; a
                tilted sample-then-filter plan passes `TILTED_IK_ABORT` so a leg that grazes the
                gate but places the piece is left for the validator to judge.

        Returns:
            The rollout: its states and actions, the phase each step belonged to, whether the
            piece was gripped and where it ended up, and what the run cost.
        """
        started = time.monotonic()
        observing = 0.0
        piece_body = self._index.piece_body_id[plan.spec.source]
        bystanders = {
            square: np.array(data.xpos[body], copy=True)
            for square, body in self._index.piece_body_id.items()
            if square != plan.spec.source
        }

        states: list[npt.NDArray[np.float64]] = []
        actions: list[npt.NDArray[np.float64]] = []
        phases: list[Phase] = []
        max_tracking = 0.0
        max_ik = 0.0
        grasped = False
        abort: AbortReason | None = None

        arm = self._arm_qpos(data)
        gripper = normalize_gripper(
            float(data.qpos[self._gripper_qadr]), self._index.gripper_joint_range
        )

        for waypoint in plan.waypoints:
            legs = max(1, round(waypoint.duration * self._control_hz))
            start_arm = arm.copy()
            start_gripper = gripper
            start_tcp = self.tcp_position(start_arm)
            target_tcp = (
                np.array(waypoint.position, dtype=np.float64)
                if waypoint.position is not None
                else None
            )
            # A Cartesian leg with an approach axis is a tilted grasp: track toward that axis
            # instead of straight down, so the wrist holds the lean the anchor was solved at
            # rather than being pulled back toward vertical and losing the IK branch. The axis is
            # eased from where the wrist starts the leg to the waypoint's, so a change of lean
            # (a source tilt handed off to a different target tilt across transport) rotates over
            # the leg rather than jumping on its first step. Legs with no axis are flown
            # fingers-down, the default every vertical grasp relies on.
            start_axis = np.array(self._approach_axis(start_arm), dtype=np.float64)
            target_axis = (
                np.array(waypoint.approach_axis, dtype=np.float64)
                if waypoint.approach_axis is not None
                else None
            )
            target_arm = (
                np.array(waypoint.qpos, dtype=np.float64) if waypoint.qpos is not None else None
            )

            for leg in range(1, legs + 1):
                blend = minimum_jerk(leg / legs)
                if target_tcp is not None:
                    point = start_tcp + blend * (target_tcp - start_tcp)
                    if target_axis is not None:
                        eased = start_axis + blend * (target_axis - start_axis)
                        self._solver._down = eased / float(np.linalg.norm(eased))
                        weight = TILT_TRACK_WEIGHT
                    else:
                        self._solver._down = self._default_down
                        weight = 1.0
                    solved, error, _ = self._solver.track(point, arm, weight=weight)
                    max_ik = max(max_ik, error)
                    if error > ik_abort:
                        abort = AbortReason.IK_DIVERGED
                    arm = solved
                else:
                    assert target_arm is not None
                    arm = start_arm + blend * (target_arm - start_arm)
                gripper = start_gripper + (leg / legs) * (waypoint.gripper - start_gripper)

                self._write_ctrl(data, arm, gripper)
                for _ in range(self._decimation):
                    mujoco.mj_step(self._model, data)

                state = self._state(data)
                states.append(state)
                actions.append(np.concatenate([arm, [gripper]]))
                phases.append(waypoint.phase)
                max_tracking = max(max_tracking, float(np.max(np.abs(state[:5] - arm))))
                if on_control_step is not None:
                    entered = time.monotonic()
                    on_control_step(len(states) - 1, data)
                    observing += time.monotonic() - entered

                grasped = grasped or (
                    waypoint.phase in (Phase.LIFT, Phase.TRANSPORT)
                    and self._holding(data, piece_body)
                )
                abort = abort or self._check_aborts(
                    data, bystanders, started + observing, wall_clock_limit
                )
                if abort is not None:
                    break
            if abort is not None:
                break

        check = self.placement(data, plan.spec.source, plan.spec.target)
        placed = abort is None and check.placed
        return ExecutionResult(
            states=np.asarray(states, dtype=np.float64).reshape(len(states), 6),
            actions=np.asarray(actions, dtype=np.float64).reshape(len(actions), 6),
            phases=tuple(phases),
            phase_boundaries=_boundaries(phases),
            grasped=grasped,
            placed=placed,
            final_square=check.final_square,
            abort_reason=abort,
            diagnostics=ExecutionDiagnostics(
                max_tracking_error=max_tracking,
                max_ik_error=max_ik,
                control_steps=len(states),
                sim_steps=len(states) * self._decimation,
                wall_seconds=time.monotonic() - started,
            ),
        )

    def placement(self, data: mujoco.MjData, source: str, target: str) -> PlacementCheck:
        """Score where the piece that started on `source` has come to rest.

        The success test the closed-loop rollout eval uses: it reads the piece's current pose and
        asks whether it is standing, upright, on `target`. This is the same final-square and
        settled check `run` applies, factored out so the policy rollout scores a move the identical
        way the oracle does — minus the abort bookkeeping a free-running rollout has no notion of.

        Args:
            data: Stepped state to read the piece's pose from.
            source: Square the piece started on, used to look its body up.
            target: Square the move was meant to end on.

        Returns:
            The resting square (or None if off-board), whether it settled, and whether that is a
            placement on `target`.
        """
        piece_body = self._index.piece_body_id[source]
        final_square = self._index.board.world_to_square(
            data.xpos[piece_body], self._index.board_pose
        )
        settled = self._settled(data, piece_body)
        return PlacementCheck(
            final_square=final_square,
            settled=settled,
            placed=final_square == target and settled,
        )

    def _settled(self, data: mujoco.MjData, piece_body: int) -> bool:
        """Report whether the moved piece is standing rather than tipped or airborne.

        Args:
            data: Stepped state to read the piece's pose from.
            piece_body: Body id of the piece that was moved.

        Returns:
            True when it sits within a centimetre of the playing surface and stands within
            10 degrees of upright.
        """
        height = float(data.xpos[piece_body][2])
        surface = self._index.board.surface_height
        upright = float(data.xmat[piece_body].reshape(3, 3)[2, 2])
        return abs(height - surface) < 0.01 and upright > float(np.cos(np.radians(10.0)))

    def _check_aborts(
        self,
        data: mujoco.MjData,
        bystanders: dict[str, npt.NDArray[np.float64]],
        deadline_from: float,
        wall_clock_limit: float,
    ) -> AbortReason | None:
        """Look for a reason to end the rollout early.

        Args:
            data: Stepped state to test.
            bystanders: Starting world position of every piece other than the one being
                moved, keyed by the square it started on.
            deadline_from: Monotonic time the budget is measured from, already pushed forward
                by whatever the caller spent outside this loop.
            wall_clock_limit: Seconds allowed past `deadline_from` before the run is aborted
                as stuck.

        Returns:
            The first condition that has been tripped, or None while the run is healthy.
        """
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            return AbortReason.NON_FINITE_STATE
        arm = self._arm_qpos(data)
        if np.any(arm < self._lo - 0.05) or np.any(arm > self._hi + 0.05):
            return AbortReason.JOINT_LIMIT
        for square, origin in bystanders.items():
            moved = float(np.linalg.norm(data.xpos[self._index.piece_body_id[square]] - origin))
            if moved > MAX_BYSTANDER_DRIFT:
                return AbortReason.PIECE_DISTURBED
        if time.monotonic() - deadline_from > wall_clock_limit:
            return AbortReason.TIMEOUT
        return None


def _boundaries(phases: list[Phase]) -> dict[Phase, tuple[int, int]]:
    """Mark out where each phase started and stopped.

    Args:
        phases: The phase every control step belonged to, in order.

    Returns:
        First and last step index of each phase that ran, keyed by phase. Phases the run
        never reached are absent.
    """
    spans: dict[Phase, tuple[int, int]] = {}
    for step, phase in enumerate(phases):
        if phase not in spans:
            spans[phase] = (step, step)
        else:
            spans[phase] = (spans[phase][0], step)
    return spans
