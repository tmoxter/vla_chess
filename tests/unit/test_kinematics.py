import time

import mujoco
import numpy as np
import pytest

from vla_chess.grasp import MAX_TCP_ERROR, load_reference_grasps
from vla_chess.kinematics import IKConfig, IKResult, IKSolver
from vla_chess.scene import ARM_JOINT_NAMES, SceneDescription, SceneIndex, build_scene

ROUNDTRIP_CASES = 500
SEED_PERTURBATION = 0.3
MIN_CONVERGENCE_RATE = 0.98
MAX_MEDIAN_ITERATIONS = 50
MAX_MEDIAN_SOLVE_SECONDS = 0.005
# The configuration limit is enforced inside the QP, so a returned angle sits on its bound to
# within floating point rather than exactly on it.
LIMIT_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def scene() -> tuple[mujoco.MjModel, SceneIndex]:
    model, index = build_scene(SceneDescription())
    return model, index


@pytest.fixture(scope="module")
def solver(scene: tuple[mujoco.MjModel, SceneIndex]) -> IKSolver:
    model, index = scene
    return IKSolver(model, index)


@pytest.fixture(scope="module")
def roundtrip(solver: IKSolver) -> list[tuple[IKResult, float]]:
    """Solve back to poses taken from known configurations, seeded off the true answer."""
    lo, hi = solver.joint_limits
    rng = np.random.default_rng(0)
    out = []
    for _ in range(ROUNDTRIP_CASES):
        truth = rng.uniform(lo, hi)
        position, quaternion = solver.fk(truth)
        seed = np.clip(truth + rng.uniform(-SEED_PERTURBATION, SEED_PERTURBATION, len(lo)), lo, hi)
        start = time.perf_counter()
        result = solver.solve(position, quaternion, seed)
        out.append((result, time.perf_counter() - start))
    return out


class TestForwardKinematics:
    def test_a_configuration_maps_to_one_pose(self, solver: IKSolver) -> None:
        qpos = np.zeros(len(ARM_JOINT_NAMES))
        first = solver.fk(qpos)
        second = solver.fk(qpos)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])

    def test_moving_a_joint_moves_the_gripper(self, solver: IKSolver) -> None:
        home = np.zeros(len(ARM_JOINT_NAMES))
        moved = home.copy()
        moved[1] = 0.4
        assert np.linalg.norm(solver.fk(home)[0] - solver.fk(moved)[0]) > 0.01

    def test_the_quaternion_is_normalized(self, solver: IKSolver) -> None:
        _, quaternion = solver.fk(np.zeros(len(ARM_JOINT_NAMES)))
        assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-9)

    def test_a_wrong_length_configuration_is_rejected(self, solver: IKSolver) -> None:
        with pytest.raises(ValueError, match="arm joint angles"):
            solver.fk(np.zeros(len(ARM_JOINT_NAMES) + 1))


class TestRoundTrip:
    def test_almost_every_reachable_pose_is_recovered(
        self, roundtrip: list[tuple[IKResult, float]]
    ) -> None:
        converged = [r for r, _ in roundtrip if r.position_error <= 0.002]
        assert len(converged) / len(roundtrip) >= MIN_CONVERGENCE_RATE

    def test_recovery_does_not_take_many_iterations(
        self, roundtrip: list[tuple[IKResult, float]]
    ) -> None:
        assert np.median([r.iterations for r, _ in roundtrip]) <= MAX_MEDIAN_ITERATIONS

    def test_a_solve_is_fast_enough_to_call_at_the_control_rate(
        self, roundtrip: list[tuple[IKResult, float]]
    ) -> None:
        assert np.median([t for _, t in roundtrip]) <= MAX_MEDIAN_SOLVE_SECONDS

    def test_no_solution_contains_a_non_finite_angle(
        self, roundtrip: list[tuple[IKResult, float]]
    ) -> None:
        assert all(np.all(np.isfinite(r.qpos)) for r, _ in roundtrip)

    def test_every_solution_respects_the_joint_limits(
        self, solver: IKSolver, roundtrip: list[tuple[IKResult, float]]
    ) -> None:
        lo, hi = solver.joint_limits
        assert all(
            np.all(r.qpos >= lo - LIMIT_TOLERANCE) and np.all(r.qpos <= hi + LIMIT_TOLERANCE)
            for r, _ in roundtrip
        )


@pytest.fixture(scope="module")
def far_away(solver: IKSolver) -> list[IKResult]:
    """Solves against targets 1.5 m from the base, well outside any configuration's reach."""
    rng = np.random.default_rng(1)
    lo, hi = solver.joint_limits
    results = []
    for _ in range(100):
        direction = rng.normal(size=3)
        target = direction / np.linalg.norm(direction) * 1.5
        results.append(solver.solve(target, np.array([1.0, 0.0, 0.0, 0.0]), rng.uniform(lo, hi)))
    return results


class TestUnreachableTargets:
    def test_a_target_out_of_reach_is_reported_as_unconverged(
        self, far_away: list[IKResult]
    ) -> None:
        assert not any(r.converged for r in far_away)

    def test_a_target_out_of_reach_still_returns_finite_angles(
        self, far_away: list[IKResult]
    ) -> None:
        assert all(np.all(np.isfinite(r.qpos)) for r in far_away)

    def test_a_target_out_of_reach_still_respects_the_joint_limits(
        self, solver: IKSolver, far_away: list[IKResult]
    ) -> None:
        lo, hi = solver.joint_limits
        assert all(
            np.all(r.qpos >= lo - LIMIT_TOLERANCE) and np.all(r.qpos <= hi + LIMIT_TOLERANCE)
            for r in far_away
        )


class TestStepCap:
    def test_no_iteration_takes_a_larger_joint_step_than_configured(self, solver: IKSolver) -> None:
        """Near full extension the unclamped least-squares step blows up; the cap is what
        keeps a diverging solve from leaving the arm somewhere unusable."""
        lo, hi = solver.joint_limits
        rng = np.random.default_rng(2)
        cap = IKConfig().max_step
        for _ in range(50):
            direction = rng.normal(size=3)
            target = direction / np.linalg.norm(direction) * 1.2
            result = solver.solve(target, np.array([1.0, 0.0, 0.0, 0.0]), hi)
            assert result.max_joint_step <= cap + 1e-12
        assert np.all(np.isfinite(lo))


class TestPostureBias:
    def test_a_solution_is_pulled_toward_the_resting_pose(
        self, scene: tuple[mujoco.MjModel, SceneIndex]
    ) -> None:
        model, index = scene
        biased = IKSolver(model, index)
        unbiased = IKSolver(model, index, IKConfig(posture_cost=0.0))
        neutral = np.array(index.arm_qpos, dtype=np.float64)
        lo, hi = biased.joint_limits
        rng = np.random.default_rng(3)

        closer = comparable = 0
        for _ in range(100):
            truth = rng.uniform(lo, hi)
            position, quaternion = biased.fk(truth)
            seed = rng.uniform(lo, hi)
            with_bias = biased.solve(position, quaternion, seed)
            without_bias = unbiased.solve(position, quaternion, seed)
            # Both are cold-started from a random seed, so a good share of them land in a
            # local minimum instead; only pairs that both solved say anything about posture.
            if not (with_bias.converged and without_bias.converged):
                continue
            comparable += 1
            if np.linalg.norm(with_bias.qpos - neutral) <= np.linalg.norm(
                without_bias.qpos - neutral
            ):
                closer += 1
        assert comparable >= 20
        assert closer / comparable >= 0.80


class TestDeterminism:
    def test_the_same_inputs_give_the_same_answer(
        self, scene: tuple[mujoco.MjModel, SceneIndex]
    ) -> None:
        model, index = scene
        first = IKSolver(model, index)
        second = IKSolver(model, index)
        rng = np.random.default_rng(4)
        lo, hi = first.joint_limits
        for _ in range(20):
            truth = rng.uniform(lo, hi)
            position, quaternion = first.fk(truth)
            seed = rng.uniform(lo, hi)
            a = first.solve(position, quaternion, seed)
            b = second.solve(position, quaternion, seed)
            assert np.array_equal(a.qpos, b.qpos)
            assert a.position_error == b.position_error
            assert a.iterations == b.iterations


class TestReferenceGrasps:
    def test_every_reference_grasp_pose_is_recovered_from_a_nearby_seed(
        self, solver: IKSolver
    ) -> None:
        """The reference set is the only ground truth for poses the arm can actually grip
        from: at the board surface, near the shoulder, and confirmed by eye. Warm-started is
        the trajectory-tracking case; the cold-start case is the slow integration test."""
        rng = np.random.default_rng(5)
        lo, hi = solver.joint_limits
        for square, grasp in sorted(load_reference_grasps().items()):
            truth = np.array(grasp.qpos, dtype=np.float64)
            position, quaternion = solver.fk(truth)
            seed = np.clip(truth + rng.uniform(-SEED_PERTURBATION, SEED_PERTURBATION, 5), lo, hi)
            result = solver.solve(position, quaternion, seed)
            assert result.converged, f"{square}: {result.position_error * 1000:.1f} mm off"
            assert result.position_error <= MAX_TCP_ERROR
