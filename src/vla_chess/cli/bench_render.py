import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import mujoco

from vla_chess.bench import (
    DEFAULT_CAMERA_COUNTS,
    DEFAULT_RESOLUTIONS,
    TARGET_RESOLUTION,
    BenchResult,
    HostInfo,
    bench_scene_path,
    read_peak_rss_bytes,
    render_budget_markdown,
    run_sweep,
    select_working_point,
)

DEFAULT_OUTPUT = Path("docs/render-budget.md")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark's command line."""
    parser = argparse.ArgumentParser(
        prog="bench-render",
        description="Measure osmesa render throughput and peak RSS, and record the working point.",
    )
    parser.add_argument("--frames", type=int, default=60, help="measured frames per case")
    parser.add_argument("--warmup", type=int, default=5, help="unmeasured frames before each case")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        help="square resolutions to sweep",
    )
    parser.add_argument(
        "--camera-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_CAMERA_COUNTS),
        help="camera counts to sweep",
    )
    parser.add_argument(
        "--target-resolution",
        type=int,
        default=TARGET_RESOLUTION,
        help="the fixed resolution the working point is evaluated at (must be in --resolutions)",
    )
    parser.add_argument("--scene", type=Path, default=None, help="MJCF scene to benchmark")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="markdown file to write")
    parser.add_argument(
        "--no-write", action="store_true", help="print the report instead of writing it"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep, report progress per case, and write the render-budget document."""
    args = parse_args(argv)
    host = HostInfo.probe()
    scene = args.scene or bench_scene_path()
    print(
        f"host: {host.cpu_count} cpu, {host.available_ram_bytes / 1e9:.2f} GB available, "
        f"mujoco {host.mujoco_version}, MUJOCO_GL={host.gl_backend}",
        flush=True,
    )
    model = mujoco.MjModel.from_xml_path(str(scene))

    def report(result: BenchResult) -> None:
        print(
            f"{result.resolution:>4}²  {result.camera_count} cam  "
            f"{result.render_fps:7.2f} render fps  {result.pipeline_fps:7.2f} pipeline fps  "
            f"peak RSS {result.peak_rss_bytes / 1e6:7.0f} MB",
            flush=True,
        )

    results = run_sweep(
        model,
        resolutions=args.resolutions,
        camera_counts=args.camera_counts,
        frames=args.frames,
        warmup=args.warmup,
        on_result=report,
    )
    working_point = select_working_point(results, target_resolution=args.target_resolution)
    print(f"working point: {working_point.rationale}", flush=True)
    print(
        f"process peak RSS over the whole sweep: {read_peak_rss_bytes() / 1e6:.0f} MB",
        flush=True,
    )

    report_text = render_budget_markdown(results, working_point, host)
    if args.no_write:
        print(report_text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report_text)
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
