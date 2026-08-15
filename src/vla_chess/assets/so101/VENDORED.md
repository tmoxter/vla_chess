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
2. Collision geometry was added to the `base` body. Upstream ships that body with visual-only
   geometry (`class="visual"`, `contype="0" conaffinity="0"`) and no collision counterpart,
   unlike every other body in the chain — a dropped object passes straight through the base.
   Each of the base's four visual parts is duplicated as a `class="collision"` mesh geom, the
   same visual/collision pairing upstream already uses for every other link (see
   `docs/reachability.md` and `tests/integration/test_scene_integration.py` for the acceptance
   check that a test object rests on it).

3. A `<contact><exclude body1="base" body2="shoulder"/></contact>` block was added, and change
   2 is not sound without it. MuJoCo filters parent-child body pairs automatically, but not
   when the parent is welded to the world — and `base` is bolted down (`dofnum` 0, `weldid`
   0). That makes `base` the one body in this chain whose collision geometry fights its own
   child: the shoulder pivots *inside* the base's cavity, and MuJoCo collides meshes as convex
   hulls, which fill that cavity. Measured over 300 sampled arm configurations, without the
   exclude the pair overlaps in 300 of them, by up to 33.9 mm, standing ~2.6e5 N of contact
   force on `shoulder_pan` and saturating that actuator at its ±3.35 N·m limit in every pose —
   the arm cannot hold the configuration it is commanded to. Every other adjacent pair in the
   arm gets this filtering for free because its parent has a joint; the shoulder's own joint
   range bounds it mechanically either way, so nothing is lost by excluding the pair.

   This is why upstream's visual-only base is not itself a bug: with no collision geometry
   there is nothing to filter. It only bites once change 2 exists.

No other geometry, joint, or actuator definition was changed. In particular the `gripper`
joint's native range (`[-0.1745, 1.7453]` rad, hinge) was left as-is: it was checked (not
assumed) to already open monotonically with increasing joint value — see
`vla_chess.scene.normalize_gripper` / `denormalize_gripper` for the `[0, 100]`
(closed/open) convention layered on top in code, per the implementation plan's requirement
to resolve this before Step 5 depends on it.
