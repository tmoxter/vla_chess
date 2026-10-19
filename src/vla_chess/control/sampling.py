"""Candidate sampling for the sample-then-filter generator: many attempts per move, from a spread.

A move is no longer planned once and trusted. It is turned into a spread of candidates — each a
grasp lean and a placement point drawn from a distribution — that are executed and judged by
physics downstream. The first candidate of every move is the clean baseline (minimum tilt, centred
placement); the rest jitter the grasp angle and the foot position so a square that only places from
some poses still yields keepers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import mujoco
import numpy as np

from vla_chess.control.planner import (
    MoveSpec,
    Plan,
    PlanQuality,
    plan_tilted_move,
)
from vla_chess.grasp.solver import MAX_TILT_DEG, VerticalGraspSolver
from vla_chess.sim.scene import SceneIndex

DEFAULT_ATTEMPTS: Final = 8


@dataclass(frozen=True)
class SampleKnobs:
    """The distribution one move's candidates are drawn from."""

    attempts: int = DEFAULT_ATTEMPTS
    tilt_jitter_deg: float = 8.0
    place_offset_radius: float = 0.004
    max_tilt_deg: float = MAX_TILT_DEG

    def __post_init__(self) -> None:
        """Reject a distribution that could not produce a candidate.

        Raises:
            ValueError: If fewer than one attempt is asked for, or a jitter is negative.
        """
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {self.attempts}")
        if self.tilt_jitter_deg < 0.0:
            raise ValueError(f"tilt_jitter_deg must be non-negative, got {self.tilt_jitter_deg}")
        if self.place_offset_radius < 0.0:
            raise ValueError(
                f"place_offset_radius must be non-negative, got {self.place_offset_radius}"
            )


@dataclass(frozen=True)
class CandidateSpec:
    """One attempt at a move: the move, the seed it was drawn with, and the sampled knobs.

    A candidate is fully described by its fields, so a run can be reproduced or logged from them
    alone. `grasp_tilt_target_deg` is None for the baseline attempt, which takes the square's
    minimum tilt; a value asks the planner for the nearest pose to that lean instead.
    """

    move: MoveSpec
    seed: int
    grasp_tilt_target_deg: float | None
    place_offset_xy: tuple[float, float]
    attempt: int


def sample_candidates(
    move: MoveSpec,
    knobs: SampleKnobs,
    base_seed: int,
    rng: np.random.Generator,
) -> list[CandidateSpec]:
    """Draw the spread of candidates to attempt for one move.

    The first candidate is the clean baseline — minimum tilt, centred placement — so a move that
    is easy is not made to rely on a lucky jitter. Each later candidate leans the grasp somewhere
    in `[0, tilt_jitter_deg]` and offsets the foot within a disk of `place_offset_radius`, and
    carries its own seed so its restart search and board jitter are independent.

    Args:
        move: The move to draw candidates for.
        knobs: The distribution to draw from.
        base_seed: Seed of the baseline candidate; attempt i gets `base_seed + i`.
        rng: Source of the sampled leans and offsets.

    Returns:
        `knobs.attempts` candidates, the baseline first.
    """
    candidates = [
        CandidateSpec(
            move=move,
            seed=base_seed,
            grasp_tilt_target_deg=None,
            place_offset_xy=(0.0, 0.0),
            attempt=0,
        )
    ]
    for attempt in range(1, knobs.attempts):
        tilt = float(rng.uniform(0.0, knobs.tilt_jitter_deg))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        distance = float(knobs.place_offset_radius * np.sqrt(rng.uniform(0.0, 1.0)))
        candidates.append(
            CandidateSpec(
                move=move,
                seed=base_seed + attempt,
                grasp_tilt_target_deg=tilt,
                place_offset_xy=(distance * np.cos(angle), distance * np.sin(angle)),
                attempt=attempt,
            )
        )
    return candidates


def plan_candidate(
    candidate: CandidateSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    solver: VerticalGraspSolver,
    max_tilt_deg: float = MAX_TILT_DEG,
) -> tuple[Plan, PlanQuality]:
    """Turn one candidate into a plan, applying its sampled lean and placement offset.

    Args:
        candidate: The attempt to plan.
        model: Scene model to solve in.
        data: State the solving poses are written into; mutated.
        index: Name-to-id lookup for the scene.
        solver: IK to search with.
        max_tilt_deg: Largest lean off vertical any anchor may hold.

    Returns:
        The plan and the quality record of its anchors.

    Raises:
        PlanInfeasible: If either square has no screened pose under the tilt cap at all.
    """
    return plan_tilted_move(
        candidate.move,
        model,
        data,
        index,
        solver,
        np.random.default_rng(candidate.seed),
        grasp_tilt_target_deg=candidate.grasp_tilt_target_deg,
        place_offset_xy=candidate.place_offset_xy,
        max_tilt_deg=max_tilt_deg,
    )
