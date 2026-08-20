import gc
from pathlib import Path

import mujoco
import pytest

from vla_chess.analysis.bench import (
    HostInfo,
    RssTracker,
    bench_cameras,
    bench_scene_path,
    read_rss_bytes,
    render_budget_markdown,
    run_sweep,
    select_working_point,
)
from vla_chess.cli.bench_render import main
from vla_chess.sensor.render import Renderer

MEMORY_TEST_FRAMES = 2000
SETTLING_FRAMES = 100
MAX_RSS_GROWTH_BYTES = 50 * 1024 * 1024

SKIP_OOM = (
    "OOM risk: rebuilds GL contexts while rendering thousands of frames, and has killed "
    "the host process on this 2-vCPU / 3.8 GB osmesa box. No pytest flag re-enables a "
    "skip — delete this marker deliberately, and only on a machine with memory headroom."
)


@pytest.fixture(scope="module")
def bench_model() -> mujoco.MjModel:
    """The benchmark chess scene compiled into an MjModel."""
    return mujoco.MjModel.from_xml_path(str(bench_scene_path()))


@pytest.mark.slow
@pytest.mark.render
class TestMemoryGrowth:
    @pytest.mark.skip(reason=SKIP_OOM)
    def test_rss_is_flat_over_2000_frames(self, bench_model: mujoco.MjModel) -> None:
        """Rendering 2000 frames grows RSS by < 50 MB measured from the 100th frame onward."""
        data = mujoco.MjData(bench_model)
        mujoco.mj_forward(bench_model, data)
        tracker = RssTracker()
        baseline = 0
        with Renderer(bench_model, bench_cameras(128, 2)) as renderer:
            for frame in range(MEMORY_TEST_FRAMES):
                mujoco.mj_step(bench_model, data)
                renderer.render(data)
                if frame == SETTLING_FRAMES:
                    gc.collect()
                    baseline = read_rss_bytes()
                    tracker = RssTracker()
                elif frame > SETTLING_FRAMES and frame % 50 == 0:
                    tracker.sample()
            tracker.sample()
        growth = tracker.peak - baseline
        assert growth < MAX_RSS_GROWTH_BYTES, (
            f"RSS grew {growth / 1e6:.1f} MB from frame {SETTLING_FRAMES} to "
            f"{MEMORY_TEST_FRAMES} (baseline {baseline / 1e6:.0f} MB)"
        )

    @pytest.mark.skip(reason=SKIP_OOM)
    def test_repeated_renderer_lifecycles_do_not_leak(self, bench_model: mujoco.MjModel) -> None:
        """Constructing and closing renderers in a sweep releases their GL contexts."""
        data = mujoco.MjData(bench_model)
        mujoco.mj_forward(bench_model, data)
        for _ in range(3):
            with Renderer(bench_model, bench_cameras(128, 2)) as renderer:
                renderer.render(data)
        gc.collect()
        baseline = read_rss_bytes()
        for _ in range(10):
            with Renderer(bench_model, bench_cameras(128, 2)) as renderer:
                renderer.render(data)
        gc.collect()
        assert read_rss_bytes() - baseline < MAX_RSS_GROWTH_BYTES


@pytest.mark.slow
@pytest.mark.render
class TestSweep:
    def test_sweep_measures_every_case_and_reports_progress(
        self, bench_model: mujoco.MjModel
    ) -> None:
        """A sweep yields one result per resolution and camera count, in order."""
        seen: list[tuple[int, int]] = []
        results = run_sweep(
            bench_model,
            resolutions=(64, 96),
            camera_counts=(1, 2),
            frames=3,
            warmup=1,
            on_result=lambda r: seen.append((r.resolution, r.camera_count)),
        )
        assert seen == [(64, 1), (64, 2), (96, 1), (96, 2)]
        assert [(r.resolution, r.camera_count) for r in results] == seen
        assert all(r.render_fps > 0 for r in results)
        assert all(r.peak_rss_bytes > 0 for r in results)

    def test_two_cameras_cost_more_render_time_than_one(self, bench_model: mujoco.MjModel) -> None:
        """The two-camera case renders strictly more work, so its throughput is lower."""
        results = run_sweep(bench_model, resolutions=(128,), camera_counts=(1, 2), frames=5)
        one, two = results
        assert two.render_fps < one.render_fps

    def test_cli_writes_a_report_naming_the_working_point(self, tmp_path: Path) -> None:
        """The entrypoint writes a render-budget document from a real measured sweep."""
        out = tmp_path / "docs" / "render-budget.md"
        exit_code = main(
            [
                "--frames",
                "3",
                "--warmup",
                "1",
                "--resolutions",
                "64",
                "--target-resolution",
                "64",
                "--out",
                str(out),
            ]
        )
        assert exit_code == 0
        text = out.read_text()
        assert "## Working point" in text
        assert "| 64 | 2 |" in text
        assert "osmesa" in text


@pytest.mark.slow
@pytest.mark.render
def test_reported_working_point_matches_the_committed_document(
    bench_model: mujoco.MjModel,
) -> None:
    """The document's structure is what a fresh sweep produces, so it can be regenerated."""
    results = run_sweep(bench_model, resolutions=(64,), camera_counts=(1, 2), frames=3)
    point = select_working_point(results, target_resolution=64)
    text = render_budget_markdown(results, point, HostInfo.probe())
    committed = Path(__file__).resolve().parents[2] / "docs" / "render-budget.md"
    assert committed.exists(), "docs/render-budget.md must be committed with measured results"
    for heading in ("## Host and toolchain", "## Method", "## Results", "## Working point"):
        assert heading in text
        assert heading in committed.read_text()
