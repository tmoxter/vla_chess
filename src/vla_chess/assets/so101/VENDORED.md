# Vendored asset: SO-101 arm

- Source: https://github.com/TheRobotStudio/SO-ARM100
- Path: `Simulation/SO101/`
- Pinned commit: `aec17bbc256d1a7342d53aaa4950595d4c30b40d`
- License: Apache-2.0 (see upstream repo `LICENSE`)

Only `so101_new_calib.xml` (the "zero at mid-range" calibration variant, per
`strategy-step-1.md` / `implementation-plan-step-1.md` §0) and the mesh STLs it
references are vendored, under `meshes/`. `so101_old_calib.xml`, the URDF variants,
`joints_properties.xml`, `scene.xml`, and the `.part` CAD source files are not needed to
build the MuJoCo scene and were not copied.

## Modifications from upstream

`so101_new_calib.xml` is not byte-identical to the upstream file:

1. `compiler/@meshdir` changed from `assets` to `meshes` to match this vendoring layout.
2. A single `<geom type="box" class="collision" .../>` was added to the `base` body. Upstream
   ships that body with visual-only geometry (`class="visual"`, `contype="0" conaffinity="0"`)
   and no collision counterpart, unlike every other body in the chain — a dropped object
   passes straight through the base. The added box is a convex proxy fitted to the combined
   mesh AABB of the base's four visual parts (see `docs/reachability.md` and
   `tests/integration/test_scene_integration.py` for the acceptance check that a test object
   rests on it).

No other geometry, joint, or actuator definition was changed. In particular the `gripper`
joint's native range (`[-0.1745, 1.7453]` rad, hinge) was left as-is: it was checked (not
assumed) to already open monotonically with increasing joint value — see
`vla_chess.scene.normalize_gripper` / `denormalize_gripper` for the `[0, 100]`
(closed/open) convention layered on top in code, per the implementation plan's requirement
to resolve this before Step 5 depends on it.
