import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from vla_chess.analysis.reachability import (
    DEFAULT_N_SAMPLES,
    DEFAULT_XY_TOLERANCE,
    DEFAULT_Z_BAND,
    STEP4_PITCH_RANGE,
    STEP4_XY_JITTER,
    STEP4_YAW_JITTER_DEG,
    probe_reachability,
    reachability_markdown,
    sample_tcp_positions,
    sweep_envelope,
)
from vla_chess.sim.scene import SceneDescription, build_scene

DEFAULT_OUTPUT = Path("docs/reachability.md")
ENVELOPE_PITCH_STEPS = 5
ENVELOPE_XY_STEPS = (-1.0, 0.0, 1.0)
ENVELOPE_YAW_STEPS = (-1.0, 0.0, 1.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the reachability probe's command line."""
    parser = argparse.ArgumentParser(
        prog="reachability-probe",
        description="Coarse FK-only sweep of which board squares the arm can reach near table"
        " height, for the default board pose and for Step 4's full randomization envelope.",
    )
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--xy-tolerance", type=float, default=DEFAULT_XY_TOLERANCE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-write", action="store_true", help="print the report instead of writing it"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Sample the arm's forward kinematics and report per-square coarse reachability, both
    for the nominal board pose and worst-case across Step 4's randomization envelope."""
    args = parse_args(argv)
    desc = SceneDescription()
    model, index = build_scene(desc)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)

    print(f"sampling {args.n_samples} joint configurations...", flush=True)
    positions = sample_tcp_positions(model, data, index, args.n_samples, rng)
    report = probe_reachability(
        index.board,
        index.board_pose,
        positions,
        xy_tolerance=args.xy_tolerance,
        z_band=DEFAULT_Z_BAND,
    )
    print(f"nominal pose: {report.reachable_count}/64 squares reachable", flush=True)
    if report.unreachable_squares:
        print(f"unreachable: {', '.join(report.unreachable_squares)}", flush=True)

    pitches = [
        STEP4_PITCH_RANGE[0]
        + i * (STEP4_PITCH_RANGE[1] - STEP4_PITCH_RANGE[0]) / (ENVELOPE_PITCH_STEPS - 1)
        for i in range(ENVELOPE_PITCH_STEPS)
    ]
    envelope = sweep_envelope(
        index.board,
        positions,
        index.board_pose.x,
        index.board_pose.y,
        index.board_pose.yaw,
        pitches=pitches,
        xy_offsets=[f * STEP4_XY_JITTER for f in ENVELOPE_XY_STEPS],
        yaws_deg=[f * STEP4_YAW_JITTER_DEG for f in ENVELOPE_YAW_STEPS],
        xy_tolerance=args.xy_tolerance,
    )
    worst_r, worst_c = envelope.worst_reachability, envelope.worst_clearance
    print(
        f"envelope worst case: {worst_r.reachable_count}/64 reachable, "
        f"{worst_c.clearance * 100:.1f} cm min clearance "
        f"({'within' if envelope.ok() else 'outside'} both floors — see docs/reachability.md "
        "for why that corner is expected and handled by per-episode rejection, not a gate here)",
        flush=True,
    )

    text = reachability_markdown(report, envelope)
    if args.no_write:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}", flush=True)
    # The envelope's worst corner is expected to violate one of the floors (implementation-
    # plan-step-1.md §4, "Board pose feasibility") — Step 4 handles that per-episode via
    # reject-and-resample, not by requiring the whole envelope to pre-clear. Only the nominal
    # single-pose result gates this command's exit code.
    return 0 if report.meets_minimum() else 1


if __name__ == "__main__":
    sys.exit(main())
