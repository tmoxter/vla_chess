import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import mujoco
import numpy as np
import numpy.typing as npt

REQUIRED_GL_BACKEND: Final = "osmesa"

RgbFrame = npt.NDArray[np.uint8]

_BACKEND_HELP: Final = (
    "This host has no GPU and no display, so 'egl' and 'glfw' cannot work; "
    f"export MUJOCO_GL={REQUIRED_GL_BACKEND} before importing vla_chess.render."
)


def check_gl_backend() -> None:
    """Raise unless the MUJOCO_GL environment variable selects the osmesa backend."""
    backend = os.environ.get("MUJOCO_GL")
    found = "unset" if backend is None else repr(backend)
    if backend != REQUIRED_GL_BACKEND:
        raise RuntimeError(
            f"MUJOCO_GL must be {REQUIRED_GL_BACKEND!r}, but it is {found}. {_BACKEND_HELP}"
        )


def _check_resolved_backend() -> None:
    """Raise unless the GL backend mujoco actually bound is osmesa."""
    import mujoco.gl_context

    resolved = mujoco.gl_context.GLContext.__module__.rsplit(".", 1)[-1]
    if resolved != REQUIRED_GL_BACKEND:
        raise RuntimeError(
            f"mujoco bound the {resolved!r} GL backend, but {REQUIRED_GL_BACKEND!r} is required. "
            "MUJOCO_GL is read once, when mujoco first initialises its GL context, so setting it "
            f"afterwards has no effect. {_BACKEND_HELP}"
        )


@dataclass(frozen=True)
class FreeCameraPose:
    """World-frame placement of a camera that is not defined in the model."""

    position: tuple[float, float, float]
    lookat: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.position == self.lookat:
            raise ValueError("camera position and lookat must differ to define a view direction")


@dataclass(frozen=True)
class CameraSpec:
    """One rendered view: an output name, a pixel resolution, and where the camera sits.

    The camera is either the model camera called `name` — optionally asserted to be attached
    to `attached_body` — or, when `pose` is given, a free camera at that world pose.
    """

    name: str
    width: int
    height: int
    fovy: float | None = None
    attached_body: str | None = None
    pose: FreeCameraPose | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera name must be a non-empty string")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"camera {self.name!r} needs a positive resolution, "
                f"got width={self.width}, height={self.height}"
            )
        if self.fovy is not None and not 0.0 < self.fovy < 180.0:
            raise ValueError(f"camera {self.name!r} fovy must lie in (0, 180), got {self.fovy}")
        if self.attached_body is not None and self.pose is not None:
            raise ValueError(
                f"camera {self.name!r}: attached_body and pose are mutually exclusive — a free "
                "camera is not attached to a body"
            )


@dataclass(frozen=True)
class _ResolvedCamera:
    """A CameraSpec paired with the camera argument `update_scene` takes for it."""

    spec: CameraSpec
    handle: Any
    free: bool


class Renderer:
    """Renders a fixed set of cameras from one MuJoCo model into reused uint8 buffers.

    One `mujoco.Renderer` (and therefore one osmesa context) is created per distinct
    resolution and shared by the cameras that use it.
    """

    def __init__(self, model: mujoco.MjModel, cameras: Sequence[CameraSpec]) -> None:
        check_gl_backend()
        _check_resolved_backend()
        specs = tuple(cameras)
        if not specs:
            raise ValueError("Renderer requires at least one camera specification")
        names = [spec.name for spec in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate camera names would collide in the output: {duplicates}")

        self._model = model
        self._specs = specs
        self._resolved = tuple(_resolve_camera(model, spec) for spec in specs)
        self._buffers: dict[str, RgbFrame] = {
            spec.name: np.empty((spec.height, spec.width, 3), dtype=np.uint8) for spec in specs
        }
        self._backends: dict[tuple[int, int], mujoco.Renderer] = {}
        self._closed = False
        try:
            for spec in specs:
                key = (spec.height, spec.width)
                if key not in self._backends:
                    self._backends[key] = mujoco.Renderer(
                        model, height=spec.height, width=spec.width
                    )
        except BaseException:
            self.close()
            raise

    @property
    def cameras(self) -> tuple[CameraSpec, ...]:
        """The camera specifications this renderer was built for, in render order."""
        return self._specs

    @property
    def camera_names(self) -> tuple[str, ...]:
        """The output keys produced by `render`, in render order."""
        return tuple(spec.name for spec in self._specs)

    @property
    def backend_count(self) -> int:
        """How many `mujoco.Renderer` instances (and osmesa contexts) are held open."""
        return len(self._backends)

    def buffer(self, name: str) -> RgbFrame:
        """The reused output buffer for one camera."""
        return self._buffers[name]

    def render(self, data: mujoco.MjData) -> dict[str, RgbFrame]:
        """Render every camera for the given state into this renderer's buffers.

        The returned arrays are owned by the renderer and are overwritten by the next call;
        callers that keep a frame must copy it.
        """
        if self._closed:
            raise RuntimeError("Renderer is closed; render() cannot be called after close()")
        for camera in self._resolved:
            spec = camera.spec
            if camera.free and spec.fovy is not None:
                self._model.vis.global_.fovy = spec.fovy
            backend = self._backends[(spec.height, spec.width)]
            backend.update_scene(data, camera=camera.handle)
            backend.render(out=self._buffers[spec.name])
        return dict(self._buffers)

    def close(self) -> None:
        """Free the GL contexts. Idempotent."""
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
        self._closed = True

    def __enter__(self) -> "Renderer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _resolve_camera(model: mujoco.MjModel, spec: CameraSpec) -> _ResolvedCamera:
    """Turn a CameraSpec into the camera argument `mujoco.Renderer.update_scene` expects."""
    if spec.pose is not None:
        return _ResolvedCamera(spec=spec, handle=_free_camera(spec.pose), free=True)

    cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, spec.name))
    if cam_id < 0:
        known = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
        raise ValueError(
            f"model has no camera named {spec.name!r}; known cameras: {known}. "
            "Give the spec a `pose` to render from a free camera instead."
        )
    if spec.attached_body is not None:
        _check_attachment(model, spec.name, cam_id, spec.attached_body)
    if spec.fovy is not None:
        model.cam_fovy[cam_id] = spec.fovy
    return _ResolvedCamera(spec=spec, handle=cam_id, free=False)


def _check_attachment(model: mujoco.MjModel, name: str, cam_id: int, body: str) -> None:
    """Raise unless the model camera is a child of the named body."""
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body))
    if body_id < 0:
        raise ValueError(f"camera {name!r} declares attached_body {body!r}, which is not in model")
    actual_id = int(model.cam_bodyid[cam_id])
    if actual_id != body_id:
        actual = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, actual_id)
        raise ValueError(
            f"camera {name!r} is attached to body {actual!r}, not to declared {body!r}"
        )


def _free_camera(pose: FreeCameraPose) -> mujoco.MjvCamera:
    """Build a free MjvCamera looking from `pose.position` at `pose.lookat`.

    Roll is fixed by MuJoCo's free-camera parameterisation: world +z stays up. A free camera's
    field of view is the model-global one, so `CameraSpec.fovy` is applied per render call.
    """
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    position = np.asarray(pose.position, dtype=np.float64)
    lookat = np.asarray(pose.lookat, dtype=np.float64)
    forward = lookat - position
    distance = float(np.linalg.norm(forward))
    forward /= distance
    camera.lookat = lookat
    camera.distance = distance
    camera.elevation = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
    camera.azimuth = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
    return camera


check_gl_backend()
