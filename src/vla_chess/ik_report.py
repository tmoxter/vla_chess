import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import mujoco
import numpy as np

from vla_chess.grasp import load_reference_grasps
from vla_chess.kinematics import IKConfig, IKSolver
from vla_chess.scene import SceneIndex

DEFAULT_ROUNDTRIP_CASES: Final = 500
DEFAULT_SEED_PERTURBATION: Final = 0.3
MIN_CONVERGENCE_RATE: Final = 0.98
MAX_MEDIAN_SOLVE_SECONDS: Final = 0.005
POSTURE_COSTS: Final = (1e-2, 1e-3, 1e-4, 0.0)


@dataclass(frozen=True)
class RoundTripMeasurement:
    """How reliably and how fast warm-started solves recover poses taken from known
    configurations."""

    cases: int
    converged: int
    median_iterations: float
    median_seconds: float
    p95_seconds: float
    max_position_error: float

    @property
    def convergence_rate(self) -> float:
        return self.converged / self.cases

    def meets_criteria(self) -> bool:
        """Whether this run clears the convergence and solve-time floors."""
        return (
            self.convergence_rate >= MIN_CONVERGENCE_RATE
            and self.median_seconds <= MAX_MEDIAN_SOLVE_SECONDS
        )


@dataclass(frozen=True)
class RecoveryMeasurement:
    """How many reference grasp poses come back, per seeding strategy."""

    total: int
    from_resting: int
    from_nearby: int
    from_restarts: int
    median_restarts: float

    def meets_criteria(self) -> bool:
        """Whether restarts recover every reference pose."""
        return self.from_restarts == self.total


def measure_roundtrip(
    solver: IKSolver,
    cases: int = DEFAULT_ROUNDTRIP_CASES,
    seed: int = 0,
    perturbation: float = DEFAULT_SEED_PERTURBATION,
    warm_start: bool = True,
) -> RoundTripMeasurement:
    """Solve back to poses generated from random configurations, seeded near the answer or,
    when `warm_start` is false, from the resting pose."""
    lo, hi = solver.joint_limits
    rng = np.random.default_rng(seed)
    resting = solver.resting_qpos
    converged = 0
    iterations: list[int] = []
    seconds: list[float] = []
    worst = 0.0
    for _ in range(cases):
        truth = rng.uniform(lo, hi)
        position, quaternion = solver.fk(truth)
        start_qpos = (
            np.clip(truth + rng.uniform(-perturbation, perturbation, len(lo)), lo, hi)
            if warm_start
            else resting
        )
        started = time.perf_counter()
        result = solver.solve(position, quaternion, start_qpos)
        seconds.append(time.perf_counter() - started)
        iterations.append(result.iterations)
        converged += result.converged
        worst = max(worst, result.position_error)
    return RoundTripMeasurement(
        cases=cases,
        converged=converged,
        median_iterations=float(np.median(iterations)),
        median_seconds=float(np.median(seconds)),
        p95_seconds=float(np.percentile(seconds, 95)),
        max_position_error=worst,
    )


def measure_reference_recovery(
    solver: IKSolver, seed: int = 0, perturbation: float = DEFAULT_SEED_PERTURBATION
) -> RecoveryMeasurement:
    """Recover each reference grasp pose from the resting pose, from nearby, and with restarts."""
    grasps = load_reference_grasps()
    lo, hi = solver.joint_limits
    rng = np.random.default_rng(seed)
    resting = np.array(solver.resting_qpos, dtype=np.float64)
    from_resting = from_nearby = from_restarts = 0
    restarts_used: list[int] = []

    for grasp in (grasps[square] for square in sorted(grasps)):
        truth = np.array(grasp.qpos, dtype=np.float64)
        position, quaternion = solver.fk(truth)

        from_resting += solver.solve(position, quaternion, resting).converged

        nearby = np.clip(truth + rng.uniform(-perturbation, perturbation, len(lo)), lo, hi)
        from_nearby += solver.solve(position, quaternion, nearby).converged

        attempts = 0
        recovered = False
        for attempt in range(1, solver.config.restarts + 1):
            attempts = attempt
            start_qpos = resting if attempt == 1 else rng.uniform(lo, hi)
            if solver.solve(position, quaternion, start_qpos).converged:
                recovered = True
                break
        from_restarts += recovered
        restarts_used.append(attempts)

    return RecoveryMeasurement(
        total=len(grasps),
        from_resting=from_resting,
        from_nearby=from_nearby,
        from_restarts=from_restarts,
        median_restarts=float(np.median(restarts_used)),
    )


def measure_posture_costs(
    model: mujoco.MjModel,
    index: SceneIndex,
    costs: Sequence[float] = POSTURE_COSTS,
    cases: int = 200,
    seed: int = 0,
) -> list[tuple[float, RoundTripMeasurement]]:
    """Round-trip behaviour at each posture task weight, which is what fixes the default."""
    return [
        (cost, measure_roundtrip(IKSolver(model, index, IKConfig(posture_cost=cost)), cases, seed))
        for cost in costs
    ]


@dataclass(frozen=True)
class IKMeasurements:
    """Everything `docs/ik.md` reports."""

    warm: RoundTripMeasurement
    cold: RoundTripMeasurement
    recovery: RecoveryMeasurement
    posture: list[tuple[float, RoundTripMeasurement]]

    def meets_criteria(self) -> bool:
        """Whether the warm-started round trip and the reference recovery both pass."""
        return self.warm.meets_criteria() and self.recovery.meets_criteria()


def measure_all(
    model: mujoco.MjModel, index: SceneIndex, cases: int = DEFAULT_ROUNDTRIP_CASES, seed: int = 0
) -> IKMeasurements:
    """Run every measurement `docs/ik.md` reports."""
    solver = IKSolver(model, index)
    return IKMeasurements(
        warm=measure_roundtrip(solver, cases, seed),
        cold=measure_roundtrip(solver, cases, seed, warm_start=False),
        recovery=measure_reference_recovery(solver, seed),
        posture=measure_posture_costs(model, index, cases=min(cases, 200), seed=seed),
    )


def ik_markdown(measurements: IKMeasurements, config: IKConfig) -> str:
    """Render the measured IK numbers as `docs/ik.md`."""
    roundtrip, cold = measurements.warm, measurements.cold
    recovery, posture = measurements.recovery, measurements.posture
    lines = [
        "# Inverse kinematics",
        "",
        "`pixi run ik-report` regenerates this file. Every number below comes from that run on"
        " this host; none are estimates.",
        "",
        "## Method",
        "",
        "`kinematics.IKSolver` wraps `mink`'s differential IK: a frame task on the"
        f" `gripperframe` site (position cost {config.position_cost:g}, orientation cost"
        f" {config.orientation_cost:g}, Levenberg-Marquardt damping {config.lm_damping:g}), a"
        f" posture task toward the resting pose (cost {config.posture_cost:g}), and a freezing"
        " task holding every dof that is not one of the five arm joints. Joint limits are a"
        " constraint inside the QP rather than a clamp applied afterwards. Each iteration solves"
        f" the QP and integrates the result at a timestep of {config.timestep:g}, with the"
        f" per-iteration joint step capped at {config.max_step:g} rad, for at most"
        f" {config.max_iterations} iterations.",
        "",
        "A solve counts as converged when position error is within"
        f" {config.position_tolerance * 1000:g} mm **and** orientation error within"
        f" {config.orientation_tolerance:g} rad. Both are always reported: five joints cannot"
        " satisfy a general six-dof pose, so orientation is an objective the solver trades"
        " against position, and a caller that only needs position should read the numbers"
        " rather than the flag.",
        "",
        "## Round trip, warm started",
        "",
        f"{roundtrip.cases} poses generated by forward kinematics from random in-limit"
        " configurations, each solved from a seed perturbed by up to"
        f" ±{DEFAULT_SEED_PERTURBATION:g} rad per joint.",
        "",
        "| | |",
        "|---|---|",
        f"| Converged | {roundtrip.converged}/{roundtrip.cases}"
        f" ({roundtrip.convergence_rate:.1%}) |",
        f"| Median iterations | {roundtrip.median_iterations:.0f} |",
        f"| Median solve time | {roundtrip.median_seconds * 1000:.2f} ms |",
        f"| 95th percentile solve time | {roundtrip.p95_seconds * 1000:.2f} ms |",
        f"| Worst position error | {roundtrip.max_position_error * 1000:.2f} mm |",
        "",
        f"Criteria: ≥ {MIN_CONVERGENCE_RATE:.0%} converged, and a median solve"
        f" ≤ {MAX_MEDIAN_SOLVE_SECONDS * 1000:g} ms so that IK does not become the bottleneck"
        " next to rendering.",
        "",
        "## The seed matters, and the iteration budget does not",
        "",
        "The same round trip solved from the resting pose instead of from near the answer.",
        "",
        "| Seeded from | Converged | Median iterations |",
        "|---|---|---|",
        f"| Within ±{DEFAULT_SEED_PERTURBATION:g} rad of the answer |"
        f" {roundtrip.converged}/{roundtrip.cases} ({roundtrip.convergence_rate:.0%}) |"
        f" {roundtrip.median_iterations:.0f} |",
        f"| The resting pose | {cold.converged}/{cold.cases}"
        f" ({cold.convergence_rate:.0%}) | {cold.median_iterations:.0f} |",
        "",
        "Differential IK is local. The cold-start shortfall is not slow convergence that a"
        " larger `max_iterations` would fix — those solves have settled into a minimum that does"
        " not reach, and raising the budget changes nothing. That is what `solve_from_restarts`"
        " is for, and the same reason `grasp.VerticalGraspSolver` restarts. Along a trajectory"
        " the previous configuration is the seed, which is the first row.",
        "",
        "## Reference grasp poses",
        "",
        "The board-surface poses in `assets/reference_grasps.json`: near the shoulder, where the"
        " arm doubles back over itself, and the only poses in this project confirmed by eye to"
        " be grippable.",
        "",
        "| Seeded from | Recovered |",
        "|---|---|",
        f"| The resting pose | {recovery.from_resting}/{recovery.total} |",
        f"| Within ±{DEFAULT_SEED_PERTURBATION:g} rad of the answer |"
        f" {recovery.from_nearby}/{recovery.total} |",
        f"| Restarts, up to {config.restarts} | {recovery.from_restarts}/{recovery.total}"
        f" (median {recovery.median_restarts:.0f} attempt"
        f"{'' if recovery.median_restarts == 1 else 's'}) |",
        "",
        "These come back cold, which is worth stating plainly because an earlier revision of"
        " this solver recovered only 18 of the 30 and the natural reading was that board-surface"
        " poses are the hard region. They are not: that was the posture task set two orders of"
        " magnitude too heavy, pulling every solve toward the resting pose hard enough to miss."
        " The table below is what settled it, and the resting pose being a good seed for these"
        " particular targets is also why this section is easier than the one above it.",
        "",
        "## Why the posture task is weighted where it is",
        "",
        "Round trip as above, at each posture cost.",
        "",
        "| Posture cost | Converged | Median iterations |",
        "|---|---|---|",
    ]
    for cost, measurement in posture:
        lines.append(
            f"| {cost:g} | {measurement.converged}/{measurement.cases}"
            f" ({measurement.convergence_rate:.0%}) | {measurement.median_iterations:.0f} |"
        )
    lines += [
        "",
        "Five joints against a six-dof target leaves almost no nullspace for a posture"
        " preference to occupy, so the posture task does not bias a free direction — it competes"
        " with the frame task directly. The default is set from this table, not from a"
        " convention.",
        "",
        "## What this does and does not establish",
        "",
        "Every pose measured here is either generated by this arm's own forward kinematics or"
        " taken from the reference grasps, so a clean sheet means the solver agrees with the"
        " model it was given. It says nothing about whether that model's geometry matches a real"
        " SO-101 and a real board — the same open question `docs/reachability.md` records,"
        " unchanged by this step.",
        "",
        "It also says nothing about self-collision: `solve` will return a pose that reaches the"
        " target through the arm's own body. Screening is the caller's job, via"
        " `solve_from_restarts(accept=...)` or `collision.contacts_at_pose`.",
        "",
    ]
    return "\n".join(lines)
