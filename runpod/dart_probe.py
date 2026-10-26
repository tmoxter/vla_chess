"""Calibrate the DART noise scale: how far does each sigma push the arm, and does the move survive?

For every Exp-1 move (one-step pushes, slot-0 funnel plan on the nominal board) the clean kept
plan is executed once, then re-executed under `ActionNoise` at each sigma for a few draws. Reports,
per sigma: the fraction of draws that still grasp and place, and how far the realized TCP strays
from the clean run's (mean and max over the episode, centimetres). No rendering — physics only.

    MUJOCO_GL=egl python runpod/dart_probe.py --sigmas 0.02,0.04,0.06,0.1 --draws 5 --workers 14
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
from vla_chess.control.executor import DART_PHASE_GAINS, TILTED_IK_ABORT, ActionNoise, Executor
from vla_chess.control.planner import MoveSpec, Phase, PlanInfeasible
from vla_chess.control.sampling import SampleKnobs, plan_candidate, sample_candidates
from vla_chess.dataset.episode import move_scene_description
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
    # the shipped schedule: noise only while the arm is high, clean descent and set-down
    "default": DART_PHASE_GAINS,
    # the first gated attempt, kept for comparison: noise fading out across the descent itself
    "fade_descend": {
        Phase.SURVEY_START: (1.0, 1.0),
        Phase.APPROACH: (1.0, 1.0),
        Phase.DESCEND: (1.0, 0.0),
        Phase.LIFT: (0.0, 1.0),
        Phase.TRANSPORT: (1.0, 0.0),
    },
}


def probe_move(job: tuple[str, str, list[float], int, float, str]) -> dict[str, object]:
    source, target, sigmas, draws, tau, schedule = job
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
    clean_tcp = np.stack([executor.tcp_position(s[:5]) for s in clean.states])
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
                make_data(model, index), plan, ik_abort=TILTED_IK_ABORT, action_noise=noise
            )
            n = min(len(run.states), len(clean_tcp))
            tcp = np.stack([executor.tcp_position(s[:5]) for s in run.states[:n]])
            dev = np.linalg.norm(tcp - clean_tcp[:n], axis=1)
            rows.append(
                {
                    "ok": run.succeeded,
                    "grasped": run.grasped,
                    "mean_cm": 100 * dev.mean(),
                    "max_cm": 100 * dev.max(),
                }
            )
        out["sigmas"][str(sigma)] = rows  # type: ignore[index]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.02,0.04,0.06,0.1,0.15")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="results/dart_probe.json")
    ap.add_argument("--schedule", default="default", choices=sorted(SCHEDULES))
    a = ap.parse_args()
    sigmas = [float(x) for x in a.sigmas.split(",")]
    moves = one_step_moves(EXP2_TRAINED_MOVES)
    jobs = [(s, t, sigmas, a.draws, a.tau, a.schedule) for s, t in moves]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(probe_move, jobs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{len(moves)} moves; clean ok: {sum(1 for r in results if r.get('clean_ok'))}")
    for sigma in sigmas:
        rows = [row for r in results if "sigmas" in r for row in r["sigmas"][str(sigma)]]  # type: ignore[index]
        ok = sum(r["ok"] for r in rows)
        print(
            f"sigma={sigma:.3f} rad: success {ok}/{len(rows)} ({ok / len(rows):.0%}), "
            f"grasped {sum(r['grasped'] for r in rows)}/{len(rows)}, "
            f"TCP dev mean {np.mean([r['mean_cm'] for r in rows]):.2f} cm, "
            f"max {np.mean([r['max_cm'] for r in rows]):.2f} cm (avg of per-run max)"
        )


if __name__ == "__main__":
    main()
