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
# fingers-down constraint stopped being satisfiable somewhere between them.
MAX_IK_ERROR: Final = 0.005

# How far a piece that is not the one being moved may shift before the episode is a failure.
MAX_BYSTANDER_DRIFT: Final = 0.005

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
        return self.abort_reason is None and self.grasped and self.placed


def minimum_jerk(alpha: float) -> float:
    """Smoothstep with zero velocity and acceleration at both ends."""
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
        return self._control_hz

    @property
    def decimation(self) -> int:
        """Physics steps per control step."""
        return self._decimation

    def tcp_position(self, qpos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """World TCP position at an arm configuration, without touching simulated state."""
        self._solver.forward(np.asarray(qpos, dtype=np.float64))
        return np.array(self._scratch.site_xpos[self._index.tcp_site_id], copy=True)

    def _arm_qpos(self, data: mujoco.MjData) -> npt.NDArray[np.float64]:
        return np.asarray(data.qpos[self._qadr], dtype=np.float64)

    def _state(self, data: mujoco.MjData) -> npt.NDArray[np.float64]:
        gripper = normalize_gripper(
            float(data.qpos[self._gripper_qadr]), self._index.gripper_joint_range
        )
        return np.concatenate([self._arm_qpos(data), [gripper]])

    def _write_ctrl(
        self, data: mujoco.MjData, arm: npt.NDArray[np.float64], gripper: float
    ) -> None:
        for name, value in zip(ARM_JOINT_NAMES, arm, strict=True):
            data.ctrl[self._index.arm_actuator_id[name]] = value
        data.ctrl[self._index.gripper_actuator_id] = denormalize_gripper(
            gripper, self._index.gripper_joint_range
        )

    def _holding(self, data: mujoco.MjData, piece_body: int) -> bool:
        """Whether both jaws are currently bearing on the piece being moved.

        One jaw is not a hold. The fixed finger rides on the `gripper` body and touches for
        reasons unrelated to gripping, so accepting either jaw reported a grasp for the run
        that left the pawn standing on its square (docs/trajectory.md).
        """
        return grip_contacts(self._model, data, piece_body).is_gripped()

    def run(
        self,
        data: mujoco.MjData,
        plan: Plan,
        on_control_step: Callable[[int, mujoco.MjData], None] | None = None,
        wall_clock_limit: float = WALL_CLOCK_LIMIT,
    ) -> ExecutionResult:
        """Execute a plan, recording one state/action pair per control step.

        `on_control_step` is called after each step with the index and the stepped state; it is
        how observation capture attaches without this loop knowing about cameras.
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
            target_arm = (
                np.array(waypoint.qpos, dtype=np.float64) if waypoint.qpos is not None else None
            )

            for leg in range(1, legs + 1):
                blend = minimum_jerk(leg / legs)
                if target_tcp is not None:
                    point = start_tcp + blend * (target_tcp - start_tcp)
                    solved, error, _ = self._solver.track(point, arm)
                    max_ik = max(max_ik, error)
                    if error > MAX_IK_ERROR:
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

        final_square = self._index.board.world_to_square(
            data.xpos[piece_body], self._index.board_pose
        )
        placed = (
            abort is None and final_square == plan.spec.target and self._settled(data, piece_body)
        )
        return ExecutionResult(
            states=np.asarray(states, dtype=np.float64).reshape(len(states), 6),
            actions=np.asarray(actions, dtype=np.float64).reshape(len(actions), 6),
            phases=tuple(phases),
            phase_boundaries=_boundaries(phases),
            grasped=grasped,
            placed=placed,
            final_square=final_square,
            abort_reason=abort,
            diagnostics=ExecutionDiagnostics(
                max_tracking_error=max_tracking,
                max_ik_error=max_ik,
                control_steps=len(states),
                sim_steps=len(states) * self._decimation,
                wall_seconds=time.monotonic() - started,
            ),
        )

    def _settled(self, data: mujoco.MjData, piece_body: int) -> bool:
        """Whether the moved piece is standing on the surface rather than tipped or airborne."""
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
    """First and last control-step index of each phase that ran."""
    spans: dict[Phase, tuple[int, int]] = {}
    for step, phase in enumerate(phases):
        if phase not in spans:
            spans[phase] = (step, step)
        else:
            spans[phase] = (spans[phase][0], step)
    return spans
