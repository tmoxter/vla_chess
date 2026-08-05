# Render budget

Measured by `pixi run bench-render` on the generation host. Regenerate this file by re-running
that task; every number below comes from that sweep, none are estimates.

## Host and toolchain

| | |
|---|---|
| Measured at | 2026-08-05T16:25:08Z |
| CPUs available to the process | 2 |
| RAM total / available at start | 4.11 GB / 2.42 GB |
| MuJoCo | 3.10.0 |
| GL backend | `MUJOCO_GL=osmesa` (software rasteriser, no GPU on this host) |

## Method

One MuJoCo scene (`src/vla_chess/assets/bench_scene.xml`: table, 8×8 board, 16 free-joint
pieces, a 5-DoF arm, two lights, an overhead `top` camera and a wrist-mounted `wrist` camera)
is loaded once. For each resolution and camera count a `vla_chess.render.Renderer` is
constructed, warmed up, and then 60 frames are measured:
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
| 128 | 1 | 6.40 | 6.33 | 156.2 | 191 MB |
| 128 | 2 | 2.64 | 2.63 | 378.5 | 215 MB |
| 224 | 1 | 5.35 | 5.29 | 187.0 | 218 MB |
| 224 | 2 | 2.19 | 2.19 | 455.8 | 222 MB |
| 256 | 1 | 4.87 | 4.83 | 205.2 | 223 MB |
| 256 | 2 | 2.05 | 2.05 | 486.9 | 224 MB |
| 320 | 1 | 4.17 | 4.14 | 239.8 | 224 MB |
| 320 | 2 | 1.78 | 1.77 | 562.3 | 226 MB |
| 512 | 1 | 2.71 | 2.70 | 369.0 | 226 MB |
| 512 | 2 | 1.24 | 1.23 | 809.5 | 232 MB |

## Working point

Resolution is fixed at 512×512 (plan §0.10,
matching SmolVLA's reference resolution) — not chosen to clear an fps threshold. The
**4.0 fps** reference point and the **peak RSS ≤ 1.60 GB** ceiling are both
still recorded: RSS is a hard physical limit on this host, fps is informational context for
Step 15's throughput calibration.

**512² is fixed regardless** — it falls short of the 4.0 fps reference point, but resolution is not traded away for throughput (§0.10).

- Resolution: **512×512** per camera (fixed).
- Measured wrist-camera render throughput: **2.71 fps**
  (369 ms per observation).
- Peak RSS at that point: **226 MB**.

512² is fixed (matches SmolVLA's reference resolution; not chosen to satisfy an fps target). Measured 2.71 fps (below the 4.0 fps reference point) at peak RSS 0.23 GB (within the 1.60 GB ceiling). The shortfall is absorbed by Step 15's episode-count and capture-fps calibration, not by reducing resolution.

## Consequences for later steps

- Dataset image size is 512×512 for
  `observation.images.wrist` (the sole camera, per §0.9).
- At 2.71 fps, rendering a 10 s episode captured at
  10 Hz costs about 37 s of render time alone; capture
  fps is the main lever left if Step 10's ≤ 120 s/episode target is missed.
- Rendering, not physics, is the bottleneck: see the render-vs-pipeline columns above.
