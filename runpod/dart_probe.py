"""Calibrate the DART noise: how far does each sigma push the arm, and does the move survive?

For every Exp-1 move (one-step pushes, slot-0 funnel plan on the nominal board) the clean kept
plan is executed once, then re-executed under `ActionNoise` at each sigma for a few draws, with the
expert gated by `PROGRESS_GATES` unless `--ungated`. Reports, per sigma: the fraction of draws that
still grasp and place, the aborts, how much longer the gates made the successful episodes, and,
per phase, how far the realized TCP sat from the commanded one (mean and max, millimetres) — the
off-path coverage the noise actually bought. No rendering — physics only.

    MUJOCO_GL=egl python runpod/dart_probe.py --sigmas 0.04,0.06,0.09 --draws 5 --workers 14
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from vla_chess.cli.generate_dataset import (
    EXP2_TRAINED_MOVES,
    SINGLE_PAWN_INSTRUCTION,
    one_step_moves,
)
from vla_chess.control.executor import (
    DART_PHASE_GAINS,
    PROGRESS_GATES,
    TILTED_IK_ABORT,
    ActionNoise,
    Executor,
)
from vla_chess.control.planner import MoveSpec, Phase, PlanInfeasible
from vla_chess.control.sampling import SampleKnobs, plan_candidate, sample_candidates
from vla_chess.dataset.episode import DartKnobs, move_scene_description
from vla_chess.dataset.validate import validate
from vla_chess.grasp.solver import VerticalGraspSolver
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    PieceType,
    build_scene,
    make_data,
)

SCHEDULES = {
    # the shipped schedule: every leg perturbed, cut down wherever the fingers are near a piece
    "default": DART_PHASE_GAINS,
    # noise only while the arm is high, faded to zero before the descent and the set-down; it
    # leaves the precision legs within ~1 mm of the expert's path
    "high_legs_only": {
        Phase.SURVEY_START: (1.0, 1.0),
        Phase.APPROACH: (1.0, 0.0),
        Phase.LIFT: (0.0, 1.0),
        Phase.TRANSPORT: (1.0, 0.0),
    },
    # the first gated attempt, kept for comparison: noise fading out across the descent itself
    "fade_descend": {
        Phase.SURVEY_START: (1.0, 1.0),
        Phase.APPROACH: (1.0, 1.0),
        Phase.DESCEND: (1.0, 0.0),
        Phase.LIFT: (0.0, 1.0),
        Phase.TRANSPORT: (1.0, 0.0),
    },
}


def probe_move(job: tuple[str, str, list[float], int, float, str, bool]) -> dict[str, object]:
    source, target, sigmas, draws, tau, schedule, gated = job
    move = MoveSpec(
        source=source,
        target=target,
        piece_type=PieceType.PAWN,
        instruction_override=SINGLE_PAWN_INSTRUCTION,
    )
    desc = move_scene_description(
        move, single_pawn_position(source), DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS
    )
    model, index = build_scene(desc)
    executor = Executor(model, index)
    knobs = SampleKnobs()
    plan = None
    for cand in sample_candidates(move, knobs, 0, np.random.default_rng(0)):
        data = make_data(model, index)
        try:
            p, quality = plan_candidate(
                cand,
                model,
                data,
                index,
                VerticalGraspSolver(model, data, index),
                max_tilt_deg=knobs.max_tilt_deg,
            )
        except PlanInfeasible:
            continue
        dry = executor.run(make_data(model, index), p, ik_abort=TILTED_IK_ABORT)
        if validate(cand, quality, dry).accepted:
            plan = p
            break
    if plan is None:
        return {"move": f"{source}{target}", "error": "no clean plan"}

    clean = executor.run(make_data(model, index), plan, ik_abort=TILTED_IK_ABORT)
    out: dict[str, object] = {
        "move": f"{source}{target}",
        "clean_ok": clean.succeeded,
        "sigmas": {},
    }
    for sigma in sigmas:
        rows = []
        for d in range(draws):
            noise = ActionNoise(
                sigma,
                tau,
                executor.control_hz,
                np.random.default_rng([7, d]),
                gains=SCHEDULES[schedule],
            )
            run = executor.run(
                make_data(model, index),
                plan,
                ik_abort=TILTED_IK_ABORT,
                action_noise=noise,
                gates=PROGRESS_GATES if gated else None,
            )
            offsets: dict[str, list[float]] = {}
            for state, action, phase in zip(run.states, run.actions, run.phases, strict=True):
                gap = executor.tcp_position(state[:5]) - executor.tcp_position(action[:5])
                offsets.setdefault(phase.value, []).append(float(np.linalg.norm(gap)))
            rows.append(
                {
                    "ok": run.succeeded,
                    "grasped": run.grasped,
                    "abort": None if run.abort_reason is None else run.abort_reason.value,
                    "extra_s": (len(run.states) - len(clean.states)) / executor.control_hz,
                    "offset_mm": {
                        k: [1000 * float(np.mean(v)), 1000 * float(np.max(v))]
                        for k, v in offsets.items()
                    },
                }
            )
        out["sigmas"][str(sigma)] = rows  # type: ignore[index]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.02,0.04,0.06,0.1,0.15")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--tau", type=float, default=DartKnobs.tau)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="results/dart_probe.json")
    ap.add_argument("--schedule", default="default", choices=sorted(SCHEDULES))
    ap.add_argument("--ungated", action="store_true", help="advance every leg on the clock")
    a = ap.parse_args()
    sigmas = [float(x) for x in a.sigmas.split(",")]
    moves = one_step_moves(EXP2_TRAINED_MOVES)
    jobs = [(s, t, sigmas, a.draws, a.tau, a.schedule, not a.ungated) for s, t in moves]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(probe_move, jobs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{len(moves)} moves; clean ok: {sum(1 for r in results if r.get('clean_ok'))}")
    for sigma in sigmas:
        rows = [row for r in results if "sigmas" in r for row in r["sigmas"][str(sigma)]]  # type: ignore[index]
        ok = [r for r in rows if r["ok"]]
        aborts: dict[str, int] = {}
        for r in rows:
            if r["abort"]:
                aborts[r["abort"]] = aborts.get(r["abort"], 0) + 1
        extra = [r["extra_s"] for r in ok]
        print(
            f"sigma={sigma:.3f} rad: success {len(ok)}/{len(rows)} ({len(ok) / len(rows):.0%}), "
            f"grasped {sum(r['grasped'] for r in rows)}/{len(rows)}, aborts {aborts}, "
            f"gate wait on success median {np.median(extra) if extra else float('nan'):.1f} s"
        )
        cells = []
        for phase in Phase:
            per = [r["offset_mm"][phase.value] for r in ok if phase.value in r["offset_mm"]]
            if per:
                cells.append(
                    f"{phase.value} {np.mean([m for m, _ in per]):.1f}/"
                    f"{np.mean([x for _, x in per]):.1f}"
                )
        print("  TCP offset mean/max mm (successful draws): " + ", ".join(cells))


if __name__ == "__main__":
    main()
