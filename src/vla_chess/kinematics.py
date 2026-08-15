"""Gripper pose to arm configuration: forward kinematics and differential IK via `mink`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import mink
import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.scene import ARM_JOINT_NAMES, TCP_SITE_NAME, SceneIndex

_QP_SOLVER: Final = "daqp"
_QP_DAMPING: Final = 1e-9
_SITE_FRAME: Final = "site"


@dataclass(frozen=True)
class IKConfig:
    """Tolerances, costs, and iteration budget for `IKSolver`."""

    position_tolerance: float = 0.002
    # Loose next to the position tolerance on purpose: five joints cannot satisfy a general
    # six-dof pose, so orientation is a weighted objective the solver trades against position
    # rather than a constraint it must meet.
    orientation_tolerance: float = 0.05
    max_iterations: int = 40
    # Integration timestep. At 1.0 the frame task's linearization is taken as a full Newton
    # step, which is what makes a warm-started solve converge in ~2 iterations; it is not a
    # simulation timestep and has nothing to do with the control rate.
    timestep: float = 1.0
    max_step: float = 0.5
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    # Three orders of magnitude below the frame task, and that gap is load-bearing rather than
    # cautious. Five joints against a six-dof target leaves almost no nullspace for a posture
    # preference to live in, so the two tasks compete directly rather than the posture one
    # picking among equally good solutions: at 1e-2 the pull toward the resting pose costs
    # enough accuracy that warm-started round trips converge 36% of the time and only 5 of the
    # 30 reference grasps come back. At 1e-3 convergence is total and the bias still shows,
    # picking the solution nearer the resting pose in 98% of converged solves. See docs/ik.md.
    posture_cost: float = 1e-3
    lm_damping: float = 1e-3
    restarts: int = 8


@dataclass(frozen=True)
class IKResult:
    """One solve: the configuration reached, how far off it is, and whether that counts."""

    qpos: npt.NDArray[np.float64]
    position_error: float
    orientation_error: float
    iterations: int
    converged: bool
    max_joint_step: float = field(default=0.0)


class IKSolver:
    """Differential IK for the gripper site, over the five arm joints.

    Local and seed-dependent: `solve` refines the configuration it is given and returns
    whatever it converges on, which from a cold seed is a local minimum that does not reach
    for around a third of reachable poses. Use `solve_from_restarts` when there is no good
    seed; along a trajectory the previous configuration is one. See docs/ik.md.
    """

    def __init__(
        self, model: mujoco.MjModel, index: SceneIndex, cfg: IKConfig | None = None
    ) -> None:
        self._model = model
        self._index = index
        self._cfg = cfg if cfg is not None else IKConfig()
        self._qadr = np.array(
            [model.jnt_qposadr[index.arm_joint_id[name]] for name in ARM_JOINT_NAMES],
            dtype=np.int64,
        )
        dofadr = [int(model.jnt_dofadr[index.arm_joint_id[name]]) for name in ARM_JOINT_NAMES]
        self._dofadr = np.array(dofadr, dtype=np.int64)
        self._lo = np.array(
            [model.jnt_range[index.arm_joint_id[n]][0] for n in ARM_JOINT_NAMES], dtype=np.float64
        )
        self._hi = np.array(
            [model.jnt_range[index.arm_joint_id[n]][1] for n in ARM_JOINT_NAMES], dtype=np.float64
        )

        # Only the arm's own joints may move. Every other dof in the scene — the gripper jaw
        # and each piece's free joint — is held by a freezing task rather than left to the
        # posture task's damping, which would drift the pieces while solving for the arm.
        frozen = [i for i in range(model.nv) if i not in set(dofadr)]
        posture_cost = np.zeros(model.nv, dtype=np.float64)
        posture_cost[self._dofadr] = self._cfg.posture_cost

        self._frame_task = mink.FrameTask(
            TCP_SITE_NAME,
            _SITE_FRAME,
            position_cost=self._cfg.position_cost,
            orientation_cost=self._cfg.orientation_cost,
            lm_damping=self._cfg.lm_damping,
        )
        self._posture_task = mink.PostureTask(model, cost=posture_cost)
        self._tasks: list[mink.tasks.task.BaseTask] = [self._frame_task, self._posture_task]
        if frozen:
            self._tasks.append(mink.DofFreezingTask(model, dof_indices=frozen))
        self._limits = [mink.ConfigurationLimit(model)]

        # The scene's resting state, used to fill the dofs the solver holds still and as the
        # posture task's neutral target. Piece placement does not enter the arm's kinematic
        # chain, so it is only ever carried along.
        self._template = np.zeros(model.nq, dtype=np.float64)
        self._template[:] = model.qpos0
        neutral = self._template.copy()
        neutral[self._qadr] = np.array(index.arm_qpos, dtype=np.float64)
        self._posture_task.set_target(neutral)
        self._configuration = mink.Configuration(model, neutral)

    @property
    def joint_limits(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Lower and upper bounds of the five arm joints, in `ARM_JOINT_NAMES` order."""
        return self._lo.copy(), self._hi.copy()

    @property
    def config(self) -> IKConfig:
        """The tolerances and costs this solver was built with."""
        return self._cfg

    @property
    def resting_qpos(self) -> npt.NDArray[np.float64]:
        """The scene's resting arm configuration, which the posture task targets."""
        return np.array(self._index.arm_qpos, dtype=np.float64)

    def fk(self, qpos: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """World position and wxyz quaternion of the gripper site at an arm configuration."""
        self._configuration.update(self._full_qpos(qpos))
        transform = self._configuration.get_transform_frame_to_world(TCP_SITE_NAME, _SITE_FRAME)
        position = np.asarray(transform.translation(), dtype=np.float64)
        quaternion = np.asarray(transform.rotation().wxyz, dtype=np.float64)
        return position, quaternion

    def solve(
        self,
        target_pos: npt.ArrayLike,
        target_quat: npt.ArrayLike,
        q_init: npt.ArrayLike,
    ) -> IKResult:
        """Refine `q_init` toward a gripper pose, reporting both errors whether or not it met
        tolerance."""
        cfg = self._cfg
        self._configuration.update(self._full_qpos(q_init))
        self._frame_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3(np.asarray(target_quat, dtype=np.float64)),
                np.asarray(target_pos, dtype=np.float64),
            )
        )

        position_error, orientation_error = self._errors()
        largest_step = 0.0
        iterations = 0
        while iterations < cfg.max_iterations:
            iterations += 1
            velocity = mink.solve_ik(
                self._configuration,
                self._tasks,
                cfg.timestep,
                _QP_SOLVER,
                damping=_QP_DAMPING,
                limits=self._limits,
            )
            step = np.asarray(velocity, dtype=np.float64) * cfg.timestep
            norm = float(np.linalg.norm(step[self._dofadr]))
            if norm > cfg.max_step:
                velocity = velocity * (cfg.max_step / norm)
                norm = cfg.max_step
            largest_step = max(largest_step, norm)
            self._configuration.integrate_inplace(velocity, cfg.timestep)
            position_error, orientation_error = self._errors()
            if (
                position_error <= cfg.position_tolerance
                and orientation_error <= cfg.orientation_tolerance
            ):
                break

        qpos = np.asarray(self._configuration.q[self._qadr], dtype=np.float64).copy()
        converged = (
            bool(np.all(np.isfinite(qpos)))
            and position_error <= cfg.position_tolerance
            and orientation_error <= cfg.orientation_tolerance
        )
        return IKResult(
            qpos=qpos,
            position_error=position_error,
            orientation_error=orientation_error,
            iterations=iterations,
            converged=converged,
            max_joint_step=largest_step,
        )

    def solve_from_restarts(
        self,
        target_pos: npt.ArrayLike,
        target_quat: npt.ArrayLike,
        rng: np.random.Generator,
        accept: Callable[[npt.NDArray[np.float64]], bool] | None = None,
    ) -> IKResult:
        """Solve without a seed, retrying from random configurations until one converges.

        `accept` screens converged solutions for anything this solver does not model, self
        collision above all; a rejected configuration is never returned. Returns the closest
        attempt when nothing converges, so the caller sees how far off the target was.
        """
        best = None
        for attempt in range(self._cfg.restarts):
            seed = (
                np.array(self._index.arm_qpos, dtype=np.float64)
                if attempt == 0
                else rng.uniform(self._lo, self._hi)
            )
            result = self.solve(target_pos, target_quat, seed)
            if result.converged and accept is not None and not accept(result.qpos):
                continue
            if best is None or result.position_error < best.position_error:
                best = result
            if result.converged:
                break
        assert best is not None
        return best

    def _full_qpos(self, arm_qpos: npt.ArrayLike) -> npt.NDArray[np.float64]:
        angles = np.asarray(arm_qpos, dtype=np.float64)
        if angles.shape != (len(ARM_JOINT_NAMES),):
            raise ValueError(
                f"expected {len(ARM_JOINT_NAMES)} arm joint angles, got shape {angles.shape}"
            )
        full = self._template.copy()
        full[self._qadr] = angles
        return full

    def _errors(self) -> tuple[float, float]:
        error = self._frame_task.compute_error(self._configuration)
        return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
