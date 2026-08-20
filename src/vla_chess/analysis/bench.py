import gc
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import mujoco

from vla_chess import paths
from vla_chess.sensor.render import REQUIRED_GL_BACKEND, CameraSpec, Renderer

MIN_CAMERA_FPS: Final = 4.0
RSS_CEILING_BYTES: Final = 1_600_000_000
TARGET_RESOLUTION: Final = 512
DEFAULT_RESOLUTIONS: Final = (128, 224, 256, 320, 512)
DEFAULT_CAMERA_COUNTS: Final = (1, 2)
PRODUCTION_CAMERA_COUNT: Final = 1
BENCH_CAMERA_NAMES: Final = ("wrist", "top")

_RSS_SAMPLE_INTERVAL: Final = 10


def bench_scene_path() -> Path:
    """Locate the scene the render budget is measured on.

    Returns:
        Path to the chess-like bench scene, resolved from the packaged asset root.
    """
    return paths.ASSETS_DIR / "bench_scene.xml"


def read_rss_bytes() -> int:
    """Read this process's current memory footprint.

    Returns:
        Resident set size in bytes, covering the interpreter and every library in it rather
        than the renderer alone.
    """
    return _read_proc_status_kib("VmRSS:") * 1024


def read_peak_rss_bytes() -> int:
    """Read this process's largest memory footprint so far.

    Returns:
        High-water mark of resident set size in bytes, as the kernel has tracked it since
        the process started.
    """
    return _read_proc_status_kib("VmHWM:") * 1024


def _read_proc_status_kib(field: str) -> int:
    """Read one kibibyte-valued field out of /proc/self/status.

    Args:
        field: Field name to match, including its trailing colon.

    Returns:
        The field's value in kibibytes.

    Raises:
        RuntimeError: If this kernel's status file carries no such field.
    """
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(field):
                return int(line.split()[1])
    raise RuntimeError(f"/proc/self/status has no {field!r} field")


class RssTracker:
    """High-water mark of resident set size over explicitly taken samples."""

    def __init__(self) -> None:
        """Start the high-water mark at the footprint the tracker is built at."""
        self._peak = read_rss_bytes()

    def sample(self) -> int:
        """Take a reading and fold it into the high-water mark.

        Returns:
            Resident set size in bytes at this moment, not the mark.
        """
        rss = read_rss_bytes()
        self._peak = max(self._peak, rss)
        return rss

    @property
    def peak(self) -> int:
        """Report the mark the samples have reached.

        Returns:
            Largest resident set size in bytes across the samples taken, which misses any
            spike between them.
        """
        return self._peak


@dataclass(frozen=True)
class HostInfo:
    """The machine and toolchain a sweep was measured on."""

    cpu_count: int
    total_ram_bytes: int
    available_ram_bytes: int
    mujoco_version: str
    gl_backend: str
    timestamp: str

    @classmethod
    def probe(cls) -> "HostInfo":
        """Read what this host and toolchain are, at the moment of the call.

        Returns:
            The host's CPU affinity count, total and available memory, MuJoCo version, GL
            backend and a UTC timestamp, so a published sweep says what produced it.
        """
        meminfo = _read_meminfo()
        return cls(
            cpu_count=len(_online_cpus()),
            total_ram_bytes=meminfo["MemTotal:"] * 1024,
            available_ram_bytes=meminfo["MemAvailable:"] * 1024,
            mujoco_version=mujoco.__version__,
            gl_backend=REQUIRED_GL_BACKEND,
            timestamp=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )


def _read_meminfo() -> dict[str, int]:
    """Read /proc/meminfo into a field -> kibibytes mapping."""
    with open("/proc/meminfo") as handle:
        fields = (line.split() for line in handle)
        return {parts[0]: int(parts[1]) for parts in fields if len(parts) >= 2}


def _online_cpus() -> list[int]:
    """CPU ids this process may run on."""
    return sorted(os.sched_getaffinity(0))


@dataclass(frozen=True)
class BenchResult:
    """Throughput and memory measured for one resolution and camera count."""

    resolution: int
    camera_count: int
    frames: int
    render_seconds: float
    physics_seconds: float
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        """Reject a result no throughput can be read out of.

        Raises:
            ValueError: If no frame was measured, or no render time was spent.
        """
        if self.frames <= 0:
            raise ValueError("frames must be positive: a result needs at least one measured frame")
        if self.render_seconds <= 0.0:
            raise ValueError("render_seconds must be positive to imply a throughput")

    @property
    def render_fps(self) -> float:
        """Measure throughput charged for rendering alone.

        Returns:
            Complete observations per second, an observation being every camera of this
            case rather than a single image.
        """
        return self.frames / self.render_seconds

    @property
    def pipeline_fps(self) -> float:
        """Measure throughput charged for rendering and physics together.

        Returns:
            Complete observations per second including the physics stepped between frames,
            at this benchmark's one step per frame rather than an episode's many.
        """
        return self.frames / (self.render_seconds + self.physics_seconds)


class RenderBudgetExceeded(MemoryError):
    """Peak RSS at the target resolution exceeds this host's hard ceiling."""


@dataclass(frozen=True)
class WorkingPoint:
    """The fixed target resolution's measured throughput and memory, and the record of it."""

    resolution: int
    camera_fps: float
    peak_rss_bytes: int
    meets_fps_reference: bool
    rationale: str


def select_working_point(
    results: Iterable[BenchResult],
    target_resolution: int = TARGET_RESOLUTION,
    min_fps: float = MIN_CAMERA_FPS,
    rss_ceiling: int = RSS_CEILING_BYTES,
    production_camera_count: int = PRODUCTION_CAMERA_COUNT,
) -> WorkingPoint:
    """Evaluate the fixed target resolution at the production camera count.

    Resolution is not chosen by searching for the highest fps-passing case — it is fixed
    regardless of throughput. RSS is still a hard, physical ceiling on this host, while
    falling short on fps is recorded rather than raised: it is a throughput fact for Step
    15's episode-count and capture-fps calibration to absorb, not a reason to shrink the
    image.

    Args:
        results: Sweep to select from. Cases at other camera counts are ignored.
        target_resolution: Square edge in pixels the working point is pinned to, matching
            SmolVLA's reference resolution.
        min_fps: Reference throughput the measured case is reported against.
        rss_ceiling: Hard memory ceiling in bytes for this host.
        production_camera_count: How many cameras the production configuration renders,
            which is the only case a working point may be read from.

    Returns:
        The target resolution's measured throughput and memory, with a rationale recording
        whether it cleared the fps reference point.

    Raises:
        ValueError: If the sweep holds no case at that resolution and camera count.
        RenderBudgetExceeded: If that case's peak RSS is over the ceiling. This is a hard
            stop rather than a fallback: nothing here silently substitutes a smaller image.
    """
    candidates = [r for r in results if r.camera_count == production_camera_count]
    match = next((r for r in candidates if r.resolution == target_resolution), None)
    if match is None:
        raise ValueError(
            f"no {target_resolution}² result at camera_count={production_camera_count}: the "
            "working point cannot be evaluated"
        )

    if match.peak_rss_bytes > rss_ceiling:
        raise RenderBudgetExceeded(
            f"{target_resolution}² peak RSS {_gb(match.peak_rss_bytes)} exceeds the "
            f"{_gb(rss_ceiling)} ceiling on this host — this is a hard stop, not a fallback "
            "case. Step 15 must explicitly recalibrate (lower resolution and/or shorter "
            "episodes) and document the change; nothing here silently substitutes a smaller "
            "image."
        )

    meets_fps = match.render_fps >= min_fps
    rationale = (
        f"{target_resolution}² is fixed (matches SmolVLA's reference resolution; not chosen "
        f"to satisfy an fps target). Measured {match.render_fps:.2f} fps "
        f"({'at or above' if meets_fps else 'below'} the {min_fps} fps reference point) at "
        f"peak RSS {_gb(match.peak_rss_bytes)} (within the {_gb(rss_ceiling)} ceiling)."
    )
    if not meets_fps:
        rationale += (
            " The shortfall is absorbed by Step 15's episode-count and capture-fps "
            "calibration, not by reducing resolution."
        )
    return WorkingPoint(
        resolution=target_resolution,
        camera_fps=match.render_fps,
        peak_rss_bytes=match.peak_rss_bytes,
        meets_fps_reference=meets_fps,
        rationale=rationale,
    )


def bench_cameras(resolution: int, camera_count: int) -> list[CameraSpec]:
    """Build the cameras one sweep case renders.

    Args:
        resolution: Square edge in pixels for every camera in the case.
        camera_count: How many cameras to render. The wrist camera comes first, since it
            alone is the production case; the overhead camera is added only to quantify
            what a second camera costs.

    Returns:
        One spec per camera, the wrist camera mounted on the wrist body and the rest fixed.

    Raises:
        ValueError: If the count is below one or above the cameras the bench scene carries.
    """
    if not 1 <= camera_count <= len(BENCH_CAMERA_NAMES):
        raise ValueError(f"camera_count must lie in [1, {len(BENCH_CAMERA_NAMES)}]")
    names = BENCH_CAMERA_NAMES[:camera_count]
    attached = {"wrist": "wrist"}
    return [
        CameraSpec(
            name=name,
            width=resolution,
            height=resolution,
            attached_body=attached.get(name),
        )
        for name in names
    ]


def measure_case(
    model: mujoco.MjModel,
    resolution: int,
    camera_count: int,
    frames: int,
    warmup: int = 5,
) -> BenchResult:
    """Time stepped-and-rendered observations at one resolution and camera count.

    The renderer is created and closed inside this call, so a sweep holds one set of GL
    contexts at a time.

    Args:
        model: Scene model to step and render. A fresh state is built for the case.
        resolution: Square edge in pixels for every camera.
        camera_count: How many cameras each observation renders.
        frames: How many measured observations to time, each one physics step and one
            render of every camera.
        warmup: How many observations to run untimed first, so first-render allocation does
            not land in the measurement.

    Returns:
        The case's render and physics time and its peak memory, which is sampled every
        tenth frame rather than continuously.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tracker = RssTracker()
    renderer = Renderer(model, bench_cameras(resolution, camera_count))
    try:
        for _ in range(warmup):
            mujoco.mj_step(model, data)
            renderer.render(data)
        tracker.sample()
        render_seconds = 0.0
        physics_seconds = 0.0
        for frame in range(frames):
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            stepped = time.perf_counter()
            renderer.render(data)
            rendered = time.perf_counter()
            physics_seconds += stepped - start
            render_seconds += rendered - stepped
            if frame % _RSS_SAMPLE_INTERVAL == 0:
                tracker.sample()
        tracker.sample()
    finally:
        renderer.close()
        del renderer
        del data
        gc.collect()
    return BenchResult(
        resolution=resolution,
        camera_count=camera_count,
        frames=frames,
        render_seconds=render_seconds,
        physics_seconds=physics_seconds,
        peak_rss_bytes=tracker.peak,
    )


def run_sweep(
    model: mujoco.MjModel,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    camera_counts: Sequence[int] = DEFAULT_CAMERA_COUNTS,
    frames: int = 60,
    warmup: int = 5,
    on_result: Callable[[BenchResult], None] | None = None,
) -> list[BenchResult]:
    """Measure every resolution and camera count in turn, one renderer alive at a time.

    Args:
        model: Scene model to step and render.
        resolutions: Square edges in pixels to measure.
        camera_counts: Camera counts to measure, each paired with every resolution.
        frames: How many measured observations to time per case.
        warmup: How many observations to run untimed before each case is timed.
        on_result: Called with each case as it finishes, for progress output. The sweep is
            long enough that waiting for the whole list is its own problem.

    Returns:
        Every case, resolution-major in the order the arguments gave them.
    """
    results: list[BenchResult] = []
    for resolution in resolutions:
        for camera_count in camera_counts:
            result = measure_case(model, resolution, camera_count, frames=frames, warmup=warmup)
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results


def render_budget_markdown(
    results: Sequence[BenchResult],
    working_point: WorkingPoint,
    host: HostInfo,
    min_fps: float = MIN_CAMERA_FPS,
    rss_ceiling: int = RSS_CEILING_BYTES,
) -> str:
    """Render a measured sweep and its working point as the body of `docs/render-budget.md`.

    Args:
        results: Every measured case, sorted into the results table.
        working_point: Evaluated target resolution, carrying the verdict and rationale.
        host: Machine and toolchain the sweep ran on, so the numbers stay attributable.
        min_fps: Reference throughput quoted alongside the verdict.
        rss_ceiling: Memory ceiling in bytes quoted alongside the verdict.

    Returns:
        The document as markdown.
    """
    rows = "\n".join(
        f"| {r.resolution} | {r.camera_count} | {r.render_fps:.2f} | {r.pipeline_fps:.2f} | "
        f"{r.render_seconds / r.frames * 1000:.1f} | {_mb(r.peak_rss_bytes)} |"
        for r in sorted(results, key=lambda r: (r.resolution, r.camera_count))
    )
    frames = sorted({r.frames for r in results})
    verdict = (
        f"**{working_point.resolution}² meets the {min_fps} fps reference point.**"
        if working_point.meets_fps_reference
        else (
            f"**{working_point.resolution}² is fixed regardless** — it falls short of the "
            f"{min_fps} fps reference point, but resolution is not traded away for "
            "throughput (§0.10)."
        )
    )
    size = f"{working_point.resolution}×{working_point.resolution}"
    return f"""# Render budget

Measured by `pixi run bench-render` on the generation host. Regenerate this file by re-running
that task; every number below comes from that sweep, none are estimates.

## Host and toolchain

| | |
|---|---|
| Measured at | {host.timestamp} |
| CPUs available to the process | {host.cpu_count} |
| RAM total / available at start | {_gb(host.total_ram_bytes)} / {_gb(host.available_ram_bytes)} |
| MuJoCo | {host.mujoco_version} |
| GL backend | `MUJOCO_GL={host.gl_backend}` (software rasteriser, no GPU on this host) |

## Method

One MuJoCo scene (`src/vla_chess/assets/bench_scene.xml`: table, 8×8 board, 16 free-joint
pieces, a 5-DoF arm, two lights, an overhead `top` camera and a wrist-mounted `wrist` camera)
is loaded once. For each resolution and camera count a `vla_chess.render.Renderer` is
constructed, warmed up, and then {" and ".join(str(f) for f in frames)} frames are measured:
each frame steps physics once and renders every camera into the renderer's reused buffers.
The renderer is closed and garbage-collected before the next case, so only one set of osmesa
contexts is alive at a time. The production configuration (plan §0.9) is the wrist camera
alone — the `Cameras` column of 2 rows are measured only to quantify the marginal cost of a
second camera, and do not decide the working point below.

*Render fps* counts complete observations (all cameras of that case) per second of render
time. *Pipeline fps* additionally charges the physics step, and is the closer analogue of
generation-time throughput — though a real episode steps physics many times per observation
and also runs IK, so end-to-end seconds-per-episode is re-measured in Step 10.

*Peak RSS* is the high-water mark of `VmRSS` sampled during the case; it includes the
Python interpreter, MuJoCo, NumPy, and the model, not just the renderer.

## Results

| Resolution (px²) | Cameras | Render fps | Pipeline fps | ms/frame (render) | Peak RSS |
|---|---|---|---|---|---|
{rows}

## Working point

Resolution is fixed at {size} (plan §0.10,
matching SmolVLA's reference resolution) — not chosen to clear an fps threshold. The
**{min_fps} fps** reference point and the **peak RSS ≤ {_gb(rss_ceiling)}** ceiling are both
still recorded: RSS is a hard physical limit on this host, fps is informational context for
Step 15's throughput calibration.

{verdict}

- Resolution: **{size}** per camera (fixed).
- Measured wrist-camera render throughput: **{working_point.camera_fps:.2f} fps**
  ({1000 / working_point.camera_fps:.0f} ms per observation).
- Peak RSS at that point: **{_mb(working_point.peak_rss_bytes)}**.

{working_point.rationale}

## Consequences for later steps

- Dataset image size is {size} for
  `observation.images.wrist` (the sole camera, per §0.9).
- At {working_point.camera_fps:.2f} fps, rendering a 10 s episode captured at
  10 Hz costs about {100 / working_point.camera_fps:.0f} s of render time alone; capture
  fps is the main lever left if Step 10's ≤ 120 s/episode target is missed.
- Rendering, not physics, is the bottleneck: see the render-vs-pipeline columns above.
"""


def _mb(size_bytes: int) -> str:
    """Format a byte count in decimal megabytes."""
    return f"{size_bytes / 1e6:.0f} MB"


def _gb(size_bytes: int) -> str:
    """Format a byte count in decimal gigabytes."""
    return f"{size_bytes / 1e9:.2f} GB"
