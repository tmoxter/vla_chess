import subprocess
import sys
import textwrap
from pathlib import Path

import mujoco
import numpy as np
import pytest

from vla_chess.sensor.render import (
    CameraSpec,
    FreeCameraPose,
    Renderer,
    check_gl_backend,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def box_model() -> mujoco.MjModel:
    """The red-box fixture scene compiled into an MjModel."""
    return mujoco.MjModel.from_xml_path(str(FIXTURES / "red_box.xml"))


@pytest.fixture
def box_data(box_model: mujoco.MjModel) -> mujoco.MjData:
    """Forward-evaluated MjData for the red-box fixture."""
    data = mujoco.MjData(box_model)
    mujoco.mj_forward(box_model, data)
    return data


def fixed_camera(width: int = 64, height: int = 64) -> CameraSpec:
    """A CameraSpec for the fixture's world-fixed camera."""
    return CameraSpec(name="fixed", width=width, height=height)


class TestBackendCheck:
    @pytest.mark.parametrize("backend", ["glfw", "", "OSMESA", "EGL"])
    def test_unsupported_backend_raises_naming_the_variable(
        self, backend: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any MUJOCO_GL value outside the supported set is rejected, naming the variable."""
        monkeypatch.setenv("MUJOCO_GL", backend)
        with pytest.raises(RuntimeError, match="MUJOCO_GL") as excinfo:
            check_gl_backend()
        assert "osmesa" in str(excinfo.value) and "egl" in str(excinfo.value)

    def test_unset_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset MUJOCO_GL is rejected rather than left to mujoco's autodetection."""
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        with pytest.raises(RuntimeError, match="MUJOCO_GL"):
            check_gl_backend()

    @pytest.mark.parametrize("backend", ["osmesa", "egl"])
    def test_supported_backend_passes(self, backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both supported backends (osmesa for CPU, egl for GPU) are accepted by the env check."""
        monkeypatch.setenv("MUJOCO_GL", backend)
        check_gl_backend()

    def test_constructing_renderer_with_unsupported_backend_raises(
        self, box_model: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Renderer construction re-checks the backend instead of trusting import time."""
        monkeypatch.setenv("MUJOCO_GL", "glfw")
        with pytest.raises(RuntimeError, match="MUJOCO_GL"):
            Renderer(box_model, [fixed_camera()])

    def test_importing_module_with_unsupported_backend_raises(self) -> None:
        """Importing vla_chess.sensor.render under an unsupported backend fails at import time."""
        script = textwrap.dedent(
            """
            import os
            os.environ["MUJOCO_GL"] = "glfw"
            try:
                import vla_chess.sensor.render  # noqa: F401
            except RuntimeError as exc:
                print(exc)
                raise SystemExit(0)
            raise SystemExit(1)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr
        assert "MUJOCO_GL" in proc.stdout


class TestCameraSpecValidation:
    @pytest.mark.parametrize(("width", "height"), [(0, 64), (64, 0), (-1, 64)])
    def test_non_positive_resolution_rejected(self, width: int, height: int) -> None:
        """A camera must have a positive pixel resolution."""
        with pytest.raises(ValueError, match="positive"):
            CameraSpec(name="c", width=width, height=height)

    def test_free_pose_and_attached_body_are_mutually_exclusive(self) -> None:
        """A camera is either a model camera or a free pose, never both."""
        pose = FreeCameraPose(position=(0.0, -1.0, 0.5), lookat=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="mutually exclusive"):
            CameraSpec(name="c", width=8, height=8, attached_body="box", pose=pose)


class TestRendererConstruction:
    def test_empty_camera_sequence_rejected(self, box_model: mujoco.MjModel) -> None:
        """A renderer with no cameras would silently produce empty observations."""
        with pytest.raises(ValueError, match="at least one camera"):
            Renderer(box_model, [])

    def test_duplicate_camera_names_rejected(self, box_model: mujoco.MjModel) -> None:
        """Duplicate names would collide in the output dict."""
        with pytest.raises(ValueError, match="duplicate"):
            Renderer(box_model, [fixed_camera(), fixed_camera(32, 32)])

    def test_unknown_model_camera_named_in_error(self, box_model: mujoco.MjModel) -> None:
        """A camera name absent from the model is reported with the known names."""
        with pytest.raises(ValueError, match="nope") as excinfo:
            Renderer(box_model, [CameraSpec(name="nope", width=8, height=8)])
        assert "fixed" in str(excinfo.value)

    def test_attached_body_mismatch_rejected(self, box_model: mujoco.MjModel) -> None:
        """A camera declared as attached to a body must actually be a child of it."""
        spec = CameraSpec(name="fixed", width=8, height=8, attached_body="box")
        with pytest.raises(ValueError, match="attached"):
            Renderer(box_model, [spec])

    def test_attached_body_match_accepted(self, box_model: mujoco.MjModel) -> None:
        """The body-attached fixture camera passes its attachment assertion."""
        spec = CameraSpec(name="mounted", width=8, height=8, attached_body="box")
        with Renderer(box_model, [spec]) as renderer:
            assert renderer.camera_names == ("mounted",)

    def test_fovy_override_written_to_model(self, box_model: mujoco.MjModel) -> None:
        """An explicit fovy overrides the model camera's compiled value."""
        cam_id = mujoco.mj_name2id(box_model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed")
        assert box_model.cam_fovy[cam_id] == pytest.approx(45.0)
        with Renderer(box_model, [CameraSpec("fixed", 8, 8, fovy=30.0)]):
            assert box_model.cam_fovy[cam_id] == pytest.approx(30.0)


@pytest.mark.render
class TestRenderOutput:
    def test_renders_requested_shape_and_dtype(
        self, box_model: mujoco.MjModel, box_data: mujoco.MjData
    ) -> None:
        """Each camera yields a uint8 (H, W, 3) array under its own name."""
        specs = [CameraSpec("fixed", width=48, height=32), CameraSpec("mounted", 24, 24)]
        with Renderer(box_model, specs) as renderer:
            frames = renderer.render(box_data)
        assert set(frames) == {"fixed", "mounted"}
        assert frames["fixed"].shape == (32, 48, 3)
        assert frames["mounted"].shape == (24, 24, 3)
        assert all(frame.dtype == np.uint8 for frame in frames.values())

    def test_red_box_dominates_red_channel(
        self, box_model: mujoco.MjModel, box_data: mujoco.MjData
    ) -> None:
        """The rendered pixels are the scene: mean red exceeds green and blue by >= 20 counts."""
        with Renderer(box_model, [fixed_camera(64, 64)]) as renderer:
            frame = renderer.render(box_data)["fixed"]
        means = frame.reshape(-1, 3).mean(axis=0)
        assert means[0] - means[1] >= 20.0
        assert means[0] - means[2] >= 20.0

    def test_free_pose_camera_sees_the_box(
        self, box_model: mujoco.MjModel, box_data: mujoco.MjData
    ) -> None:
        """A camera specified by a world pose renders the scene without a model camera."""
        pose = FreeCameraPose(position=(0.0, -0.6, 0.0), lookat=(0.0, 0.0, 0.0))
        spec = CameraSpec(name="free", width=64, height=64, fovy=45.0, pose=pose)
        with Renderer(box_model, [spec]) as renderer:
            frame = renderer.render(box_data)["free"]
        means = frame.reshape(-1, 3).mean(axis=0)
        assert means[0] - means[1] >= 20.0

    def test_consecutive_renders_are_bit_identical(
        self, box_model: mujoco.MjModel, box_data: mujoco.MjData
    ) -> None:
        """Rendering the same MjData twice produces identical pixels."""
        with Renderer(box_model, [fixed_camera()]) as renderer:
            first = renderer.render(box_data)["fixed"].copy()
            second = renderer.render(box_data)["fixed"].copy()
        assert np.array_equal(first, second)

    def test_translating_the_box_changes_at_least_one_percent_of_pixels(
        self, box_model: mujoco.MjModel, box_data: mujoco.MjData
    ) -> None:
        """A 5 cm translation of the box is visible in >= 1% of pixels."""
        with Renderer(box_model, [fixed_camera()]) as renderer:
            before = renderer.render(box_data)["fixed"].copy()
            box_data.qpos[0] = 0.05
            mujoco.mj_forward(box_model, box_data)
            after = renderer.render(box_data)["fixed"]
            changed = np.any(before != after, axis=-1)
        assert changed.mean() >= 0.01

    def test_output_buffers_are_reused_across_calls(
        self, box_model: mujoco.MjModel, box_data: mujoco.MjData
    ) -> None:
        """render() writes into buffers owned by the renderer, allocating nothing per frame."""
        with Renderer(box_model, [fixed_camera()]) as renderer:
            first = renderer.render(box_data)
            second = renderer.render(box_data)
            assert first["fixed"] is second["fixed"]
            assert first["fixed"] is renderer.buffer("fixed")

    def test_render_after_close_raises(
        self, box_model: mujoco.MjModel, box_data: mujoco.MjData
    ) -> None:
        """A closed renderer refuses to render rather than using freed GL resources."""
        renderer = Renderer(box_model, [fixed_camera()])
        renderer.close()
        with pytest.raises(RuntimeError, match="closed"):
            renderer.render(box_data)

    def test_close_is_idempotent(self, box_model: mujoco.MjModel) -> None:
        """Closing twice is safe so cleanup paths can be unconditional."""
        renderer = Renderer(box_model, [fixed_camera()])
        renderer.close()
        renderer.close()

    def test_same_resolution_cameras_share_one_backend(self, box_model: mujoco.MjModel) -> None:
        """Cameras of equal resolution reuse a single mujoco.Renderer and GL context."""
        specs = [CameraSpec("fixed", 32, 32), CameraSpec("mounted", 32, 32)]
        with Renderer(box_model, specs) as renderer:
            assert renderer.backend_count == 1
        specs = [CameraSpec("fixed", 32, 32), CameraSpec("mounted", 16, 16)]
        with Renderer(box_model, specs) as renderer:
            assert renderer.backend_count == 2
