import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vla_chess.analysis.ik_report import DEFAULT_ROUNDTRIP_CASES, ik_markdown, measure_all
from vla_chess.analysis.reference_ik import IKConfig
from vla_chess.sim.scene import SceneDescription, build_scene

DEFAULT_OUTPUT = Path("docs/ik.md")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the IK report's command line.

    Args:
        argv: Command line to parse, without the program name. `sys.argv` is read when
            omitted.

    Returns:
        The parsed options: how many round trips to solve, the seed to solve them from, where
        to write the report, and whether to print it instead.
    """
    parser = argparse.ArgumentParser(
        prog="ik-report",
        description="Measure the differential IK solver's convergence, speed, and seed"
        " sensitivity on this host, and write docs/ik.md.",
    )
    parser.add_argument("--cases", type=int, default=DEFAULT_ROUNDTRIP_CASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-write", action="store_true", help="print the report instead of writing it"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the IK solver and report convergence, timing, and seed sensitivity.

    Args:
        argv: Command line to parse, without the program name. `sys.argv` is read when
            omitted.

    Returns:
        Process exit status: 0 if the measurements meet the solver's criteria, 1 if they do
        not — the report is written either way.
    """
    args = parse_args(argv)
    model, index = build_scene(SceneDescription())

    print(f"solving {args.cases} round trips and the reference grasp set...", flush=True)
    measurements = measure_all(model, index, args.cases, args.seed)
    warm, cold, recovery = measurements.warm, measurements.cold, measurements.recovery

    print(
        f"round trip: {warm.converged}/{warm.cases} converged warm started "
        f"({cold.converged}/{cold.cases} from the resting pose), "
        f"median {warm.median_seconds * 1000:.2f} ms, "
        f"{warm.median_iterations:.0f} iterations",
        flush=True,
    )
    print(
        f"reference grasps: {recovery.from_resting}/{recovery.total} from the resting pose, "
        f"{recovery.from_nearby}/{recovery.total} warm started, "
        f"{recovery.from_restarts}/{recovery.total} with restarts",
        flush=True,
    )

    text = ik_markdown(measurements, IKConfig())
    if args.no_write:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}", flush=True)

    return 0 if measurements.meets_criteria() else 1


if __name__ == "__main__":
    sys.exit(main())
