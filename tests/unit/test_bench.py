from pathlib import Path

import pytest

from vla_chess.bench import (
    MIN_CAMERA_FPS,
    RSS_CEILING_BYTES,
    TARGET_RESOLUTION,
    BenchResult,
    HostInfo,
    RenderBudgetExceeded,
    RssTracker,
    read_rss_bytes,
    render_budget_markdown,
    select_working_point,
)

GIB = 1024**3
COMMITTED_REPORT = Path(__file__).resolve().parents[2] / "docs" / "render-budget.md"


def result(
    resolution: int,
    camera_count: int,
    render_fps: float,
    peak_rss_bytes: int = 400_000_000,
    frames: int = 100,
) -> BenchResult:
    """A BenchResult with the timing implied by a wanted frames-per-second."""
    return BenchResult(
        resolution=resolution,
        camera_count=camera_count,
        frames=frames,
        render_seconds=frames / render_fps,
        physics_seconds=frames / 1000.0,
        peak_rss_bytes=peak_rss_bytes,
    )


HOST = HostInfo(
    cpu_count=2,
    total_ram_bytes=4 * GIB,
    available_ram_bytes=2 * GIB,
    mujoco_version="3.2.0",
    gl_backend="osmesa",
    timestamp="2026-08-05T12:00:00Z",
)


class TestBenchResult:
    def test_render_fps_is_frames_over_render_seconds(self) -> None:
        """Render throughput excludes the physics time measured alongside it."""
        res = BenchResult(
            resolution=128,
            camera_count=1,
            frames=50,
            render_seconds=2.0,
            physics_seconds=0.5,
            peak_rss_bytes=1,
        )
        assert res.render_fps == pytest.approx(25.0)

    def test_pipeline_fps_includes_physics(self) -> None:
        """Pipeline throughput is the frames over the total stepped-and-rendered time."""
        res = BenchResult(
            resolution=128,
            camera_count=1,
            frames=50,
            render_seconds=2.0,
            physics_seconds=0.5,
            peak_rss_bytes=1,
        )
        assert res.pipeline_fps == pytest.approx(20.0)

    def test_zero_frames_rejected(self) -> None:
        """A result with no frames would report a meaningless throughput."""
        with pytest.raises(ValueError, match="frames"):
            BenchResult(
                resolution=128,
                camera_count=1,
                frames=0,
                render_seconds=1.0,
                physics_seconds=0.0,
                peak_rss_bytes=1,
            )


class TestRssTracker:
    def test_read_rss_is_a_plausible_positive_size(self) -> None:
        """The process reports a resident set larger than a megabyte and smaller than a host."""
        rss = read_rss_bytes()
        assert 1_000_000 < rss < 64 * GIB

    def test_peak_tracks_the_largest_sample(self) -> None:
        """The tracker keeps the high-water mark across samples, not the latest value."""
        tracker = RssTracker()
        baseline = tracker.peak
        ballast = bytearray(200 * 1024 * 1024)
        tracker.sample()
        high = tracker.peak
        del ballast
        tracker.sample()
        assert high > baseline
        assert tracker.peak == high


class TestWorkingPointSelection:
    def test_default_target_is_512(self) -> None:
        """The production resolution is fixed at 512², matching SmolVLA's reference input."""
        assert TARGET_RESOLUTION == 512

    def test_resolution_is_fixed_regardless_of_fps(self) -> None:
        """The target resolution is evaluated as-is — never swapped for a faster one."""
        results = [result(128, 1, 30.0), result(256, 1, 1.0)]
        point = select_working_point(results, target_resolution=256)
        assert point.resolution == 256
        assert not point.meets_fps_reference

    def test_rss_over_ceiling_is_a_hard_stop(self) -> None:
        """RSS is a physical ceiling on this host — exceeding it raises, it is not absorbed."""
        results = [result(256, 1, 9.0, peak_rss_bytes=RSS_CEILING_BYTES + 1)]
        with pytest.raises(RenderBudgetExceeded, match="exceeds"):
            select_working_point(results, target_resolution=256)

    def test_rss_within_ceiling_is_reported_not_raised(self) -> None:
        """Staying under the ceiling is unremarkable: no exception, just a recorded number."""
        results = [result(256, 1, 9.0, peak_rss_bytes=RSS_CEILING_BYTES - 1)]
        point = select_working_point(results, target_resolution=256)
        assert point.peak_rss_bytes == RSS_CEILING_BYTES - 1

    def test_two_camera_results_do_not_decide_the_working_point(self) -> None:
        """The criterion is production (1-camera) throughput; a 2-camera row is context only."""
        results = [result(128, 2, 30.0), result(128, 1, 1.0)]
        point = select_working_point(results, target_resolution=128)
        assert point.camera_fps == pytest.approx(1.0)

    def test_fps_shortfall_is_recorded_not_resolved_by_changing_resolution(self) -> None:
        """Missing the fps reference point is recorded, not fixed by picking another resolution."""
        results = [result(320, 1, 0.5)]
        point = select_working_point(results, target_resolution=320)
        assert point.resolution == 320
        assert not point.meets_fps_reference
        assert str(MIN_CAMERA_FPS) in point.rationale

    def test_rationale_names_the_binding_constraint_when_fps_is_short(self) -> None:
        """The recorded decision says which criterion failed, so it can be re-checked later."""
        results = [result(128, 1, 2.0)]
        point = select_working_point(results, target_resolution=128)
        assert "fps" in point.rationale.lower()

    def test_no_production_camera_results_rejected(self) -> None:
        """A sweep without the production (1-camera) case cannot decide the working point."""
        with pytest.raises(ValueError, match="camera_count=1"):
            select_working_point([result(128, 2, 20.0)], target_resolution=128)

    def test_no_matching_resolution_rejected(self) -> None:
        """A sweep that never measured the target resolution cannot decide the working point."""
        with pytest.raises(ValueError, match="512"):
            select_working_point([result(128, 1, 20.0)])


class TestMarkdownReport:
    def test_report_contains_every_measured_case(self) -> None:
        """Every resolution and camera count measured appears in the committed table."""
        results = [
            result(128, 1, 20.0),
            result(128, 2, 12.0),
            result(224, 1, 9.0),
            result(224, 2, 5.0),
        ]
        point = select_working_point(results, target_resolution=224)
        text = render_budget_markdown(results, point, HOST)
        for line in ("128", "224"):
            assert line in text
        assert text.count("| 128 | 1 |") == 1
        assert text.count("| 224 | 2 |") == 1

    def test_report_records_the_working_point_and_the_criteria(self) -> None:
        """The doc states the chosen point and the thresholds it was judged against."""
        results = [result(128, 1, 12.0), result(224, 1, 5.0)]
        point = select_working_point(results, target_resolution=224)
        text = render_budget_markdown(results, point, HOST)
        assert "## Working point" in text
        assert "224" in text
        assert "1.6" in text
        assert str(MIN_CAMERA_FPS) in text

    def test_report_marks_a_missed_fps_reference_explicitly(self) -> None:
        """Falling short of the fps reference point is written down, not silently absorbed."""
        results = [result(224, 1, 0.5)]
        point = select_working_point(results, target_resolution=224)
        text = render_budget_markdown(results, point, HOST)
        assert "falls short" in text.lower() or "does not" in text.lower()
        assert "224" in text

    def test_resolution_is_spelled_one_way_throughout(self) -> None:
        """One spelling of the resolution, so a regenerated doc is a clean diff of the numbers."""
        results = [result(512, 1, 2.5)]
        text = render_budget_markdown(results, select_working_point(results), HOST)
        assert "512×512" in text
        assert "512x512" not in text

    def test_committed_document_is_generator_output(self) -> None:
        """The committed doc is regenerated by the task, never hand-edited into a variant form."""
        committed = COMMITTED_REPORT.read_text()
        assert "512×512" in committed
        assert "512x512" not in committed

    def test_report_records_host_and_backend(self) -> None:
        """Numbers are only meaningful with the host and GL backend they were measured on."""
        results = [result(128, 1, 12.0)]
        point = select_working_point(results, target_resolution=128)
        text = render_budget_markdown(results, point, HOST)
        assert "osmesa" in text
        assert "3.2.0" in text
        assert "2026-08-05T12:00:00Z" in text
